import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import Subset, DataLoader
from data_handler import DataHandler
from models import PreActResNet18

def parse_args():
    parser = argparse.ArgumentParser(description='Train MixMo models')
    parser.add_argument('--dataset', type=str, default='tinyimagenet')
    parser.add_argument('--approach', type=str, required=True, 
                      choices=['linear_mixmo','cut_mixmo'])
    parser.add_argument('--width', type=int, default=1,choices = [1,2,3])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_repetitions', type=int, default=2, help='Batch repetition factor (b parameter)')
    parser.add_argument('--epochs', type=int, default=None, help='If None, will use approach-specific default')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter for Beta distribution')
    parser.add_argument('--data_root', type=str, default='/home/ahmedyra/projects/def-hinat/ahmedyra/MixMo_Replication-main/data')
    parser.add_argument('--save_dir', type=str, default='/home/ahmedyra/projects/def-hinat/ahmedyra/MixMo_Replication-main/PreAct_ImagrNet_Results')
    parser.add_argument('--run_number', type=int, default=1, help='Run 1, 2, or 3 for averaging')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def get_model(args, num_classes):
    """Create model based on specified approach"""
    aug_type = 'none'
    if args.approach in ['linear_mixmo',]:
        aug_type = 'LinearMixMo'
    elif args.approach in ['cut_mixmo']:
        aug_type = 'CutMixMo'
    
    return PreActResNet18(
        widen_factor=args.width,
        num_classes=num_classes,
        augmentation_type=aug_type,
    )

def mixmo_loss(outputs, targets, kappa=None, r=3.0):
    """MixMo loss weighting scheme from equation (3)"""
    out1, out2, out_mix1, out_mix2 = outputs
    y1, y2 = targets
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Calculate base losses
    loss1 = criterion(out_mix1, y1)
    loss2 = criterion(out_mix2, y2)
    
    if kappa is not None:
        # Apply weighting function from equation (3)
        weight1 = 2 * (kappa ** (1/r)) / ((kappa ** (1/r)) + ((1-kappa) ** (1/r)))
        weight2 = 2 * ((1-kappa) ** (1/r)) / ((kappa ** (1/r)) + ((1-kappa) ** (1/r)))
        
        # Apply weights to each example in the batch
        loss1 = (loss1 * weight1).mean()
        loss2 = (loss2 * weight2).mean()
    else:
        loss1 = loss1.mean()
        loss2 = loss2.mean()
    
    return loss1 + loss2

class TemperatureScaling(nn.Module):
    """
    A module to perform temperature scaling for model calibration.
    """
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
    def forward(self, logits):
        """
        Scale the logits by the temperature parameter
        """
        return logits / self.temperature

def optimize_temperature(model, val_loader, device, args):
    """
    Find the optimal temperature for calibration
    """
    # Set up temperature scaling model
    temperature_model = TemperatureScaling().to(device)
    
    # Set up NLL loss
    nll_criterion = nn.CrossEntropyLoss()
    
    # Optimizer for temperature parameter
    optimizer = optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)
    
    # Collect all logits and labels from validation set
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        model.eval()
        for inputs, targets in tqdm(val_loader, desc="Collecting logits for calibration"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # For inference, use the same input for both branches
            out1, out2, _, _, _ = model(inputs, inputs)
            
            # Ensemble prediction (average of both outputs)
            ensemble_output = (out1 + out2) / 2
            logits_list.append(ensemble_output)
            labels_list.append(targets)
    
    # Concatenate all logits and labels
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    # Define the optimization step
    def eval():
        optimizer.zero_grad()
        scaled_logits = temperature_model(logits)
        loss = nll_criterion(scaled_logits, labels)
        loss.backward()
        return loss
    
    # Optimize temperature
    optimizer.step(eval)
    
    return temperature_model.temperature.item()

def calculate_nll(logits, targets):
    """
    Calculate the negative log-likelihood (NLL)
    """
    # Apply softmax to convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=1)
    
    # Get the probability of the correct class for each sample
    correct_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()
    
    # Calculate the negative log likelihood
    nll = -torch.log(correct_probs)
    
    return nll.mean().item()

def train_epoch(model, train_loader, optimizer, device, args, epoch=0, total_epochs=None):
    """Unified training function for all approaches"""
    model.train()
    running_loss = 0.0
    mixing_prob = 0.5  # Default mixing probability
    
    # Calculate when to start decreasing probability (last 1/12 of training)
    if args.approach in ['cut_mixmo'] and total_epochs is not None:
        decay_start = total_epochs - total_epochs // 12
        
        if epoch >= decay_start:
            # Linear decay from 0.5 to 0 over the last 1/12 epochs
            progress = (epoch - decay_start) / (total_epochs - decay_start)
            mixing_prob = 0.5 * (1 - progress)
    
    # MixMo training
    batch_repetitions = args.batch_repetitions
    
    correct1 = 0
    correct2 = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        # Reshape to get the repeated samples
        batch_size = inputs.size(0) // batch_repetitions
        x_chunks = torch.chunk(inputs, batch_repetitions)
        y_chunks = torch.chunk(targets, batch_repetitions)
        
        # Take first two chunks for MixMo
        x1, x2 = x_chunks[0].to(device), x_chunks[1].to(device)
        y1, y2 = y_chunks[0].to(device), y_chunks[1].to(device)
        
        # For Cut-MixMo, decide whether to use patch mixing or linear mixing
        use_patch_mixing = True  # Default for Cut-MixMo is patch mixing
        if args.approach in ['cut_mixmo']:
            use_patch_mixing = torch.rand(1).item() < mixing_prob
        
        # Set mixing mode before forward pass
        if hasattr(model, 'set_mixing_mode'):
            model.set_mixing_mode('patch' if use_patch_mixing else 'linear')
        
        # Forward pass
        out1, out2, out_mix1, out_mix2, kappa = model(x1, x2)
        
        # Calculate loss
        loss = mixmo_loss((out1, out2, out_mix1, out_mix2), (y1, y2), kappa, r=3.0)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Handle CutMix targets which might be one-hot encoded
        # Get the original class labels
        if isinstance(y1, torch.Tensor) and y1.dim() > 1 and y1.size(1) > 1:
            # One-hot encoded targets from CutMix - get the class indices
            true_y1 = y1.argmax(dim=1)
        else:
            # Regular class indices
            true_y1 = y1
            
        if isinstance(y2, torch.Tensor) and y2.dim() > 1 and y2.size(1) > 1:
            # One-hot encoded targets from CutMix - get the class indices
            true_y2 = y2.argmax(dim=1)
        else:
            # Regular class indices
            true_y2 = y2
        
        # Calculate accuracy for individual subnetworks
        _, predicted1 = out1.max(1)
        correct1 += predicted1.eq(true_y1).sum().item()
        
        _, predicted2 = out2.max(1)
        correct2 += predicted2.eq(true_y2).sum().item()
        
        total += batch_size
    
    acc1 = 100. * correct1 / total
    acc2 = 100. * correct2 / total
    avg_acc = (acc1 + acc2) / 2
    
    return running_loss / len(train_loader), avg_acc, acc1, acc2

def evaluate(model, test_loader, device, args, calculate_metrics=True, optimal_temp=None):
    """Evaluation function for all approaches with NLLc calculation"""
    model.eval()
    
    # MixMo evaluation
    correct_ensemble = 0
    correct1 = 0
    correct2 = 0
    correct_top5 = 0
    total = 0
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # For inference, use the same input for both branches
            out1, out2, _, _, _ = model(inputs, inputs)
            
            # Ensemble prediction (average of both outputs)
            ensemble_output = (out1 + out2) / 2
            
            # Save logits and targets for NLL calculation
            if calculate_metrics:
                all_logits.append(ensemble_output.detach().cpu())
                all_targets.append(targets.detach().cpu())
            
            # Handle targets that might be one-hot encoded
            if isinstance(targets, torch.Tensor) and targets.dim() > 1 and targets.size(1) > 1:
                # For one-hot encoded labels from CutMix
                true_targets = targets.argmax(dim=1)
            else:
                # For class indices
                true_targets = targets
            
            # Top-1 accuracy
            _, predicted_ensemble = ensemble_output.max(1)
            correct_ensemble += predicted_ensemble.eq(true_targets).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = ensemble_output.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            top5_correct = top5_pred.eq(true_targets.view(1, -1).expand_as(top5_pred))
            correct_top5 += top5_correct.sum().item()
            
            # Subnetwork accuracies
            _, predicted1 = out1.max(1)
            correct1 += predicted1.eq(true_targets).sum().item()
            
            _, predicted2 = out2.max(1)
            correct2 += predicted2.eq(true_targets).sum().item()
            
            total += targets.size(0)
    
    acc_ensemble = 100. * correct_ensemble / total
    acc_top5 = 100. * correct_top5 / total
    acc1 = 100. * correct1 / total
    acc2 = 100. * correct2 / total
    
    # Calculate NLLc if needed
    nllc = None
    if calculate_metrics and optimal_temp is not None:
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        
        # Convert targets to class indices if they're one-hot
        if all_targets.dim() > 1 and all_targets.size(1) > 1:
            all_targets = all_targets.argmax(dim=1)
        
        # Apply temperature scaling
        scaled_logits = all_logits / optimal_temp
        
        # Calculate NLLc
        nllc = calculate_nll(scaled_logits, all_targets)
        
    return acc_ensemble, acc_top5, acc1, acc2, nllc

def main():
    args = parse_args()
    
    # Set seed based on run number
    seed = args.seed + args.run_number - 1
    set_seed(seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataset specific parameters
    num_classes = 200 if args.dataset == 'tinyimagenet' 
    
    args.epochs = 1000
    
    weight_decay = 1e-4
    
    # Setup data handler
    data_handler = DataHandler(data_root=args.data_root)
    
    # Configure data loaders based on approach
    batch_repetitions = args.batch_repetitions
    print(f"Using batch repetition: b = {batch_repetitions}")
        

        # For non-CutMix approaches, use the regular loading with repetitions
    if args.dataset == 'tinyimagenet':
        train_loader, test_loader = data_handler.get_tiny_imagenet(
            batch_size=args.batch_size * batch_repetitions, 
            batch_repetitions=batch_repetitions
        )

    
    # Create and configure model
    model = get_model(args, num_classes).to(device)

    initial_lr = 0.2 / args.batch_repetitions
    optimizer = optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[600, 900, 1200], gamma=0.1)
    # Generate experiment name
    exp_name = f"{args.approach}_b{batch_repetitions}_wrn{args.width}_{args.dataset}_run{args.run_number}"
    print(f"Starting experiment: {exp_name}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Weight decay: {weight_decay}")
    
    # Initialize results storage
    results = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_top5_acc': [],
        'test_acc_subnet1': [],
        'test_acc_subnet2': [],
        'test_nllc': []  # Adding NLLc tracking
    }
    
    best_acc = 0.0
    best_epoch = 0
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        # Train the model
        train_loss, train_acc, train_acc1, train_acc2 = train_epoch(
            model, train_loader, optimizer, device, args, 
            epoch=epoch, total_epochs=args.epochs
        )
        scheduler.step()
        
        # Regular evaluation without NLLc for training monitoring
        test_acc, test_top5_acc, test_acc1, test_acc2, _ = evaluate(
            model, test_loader, device, args,
            calculate_metrics=False
        )
        
        # Log regular results
        results['epoch'].append(epoch)
        results['train_loss'].append(float(train_loss))
        results['train_acc'].append(float(train_acc))
        results['test_acc'].append(float(test_acc))
        results['test_top5_acc'].append(float(test_top5_acc))
        results['test_acc_subnet1'].append(float(test_acc1))
        results['test_acc_subnet2'].append(float(test_acc2))
        
        # For now, just add None for NLLc during training
        results['test_nllc'].append(None)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Acc: {test_acc:.2f}%, Test Top-5 Acc: {test_top5_acc:.2f}%")
        print(f"Test Acc Subnet1: {test_acc1:.2f}%, Subnet2: {test_acc2:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            
            # Save model
            model_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
    
    # After training, split test set in half for temperature calibration
    test_indices = list(range(len(test_loader.dataset)))
    np.random.shuffle(test_indices)
    split_index = len(test_indices) // 2
    
    # Create calibration and evaluation datasets
    calib_dataset = Subset(test_loader.dataset, test_indices[:split_index])
    eval_dataset = Subset(test_loader.dataset, test_indices[split_index:])
    
    # Create corresponding data loaders
    calib_loader = DataLoader(
        calib_dataset, 
        batch_size=args.batch_size,
        shuffle=False
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Load the best model
    model_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Optimize temperature on calibration set
    print("Optimizing temperature for calibration...")
    optimal_temp = optimize_temperature(model, calib_loader, device, args)
    print(f"Optimal temperature: {optimal_temp:.4f}")
    
    # Final evaluation with temperature scaling on evaluation set
    test_acc, test_top5_acc, test_acc1, test_acc2, nllc = evaluate(
       model, eval_loader, device, args,
       optimal_temp=optimal_temp
    )
           
    # Update results with final NLLc value
    if nllc is not None:
       results['final_nllc'] = float(nllc)
       print(f"Final NLLc: {nllc:.4f}")
    
    # Save final results
    results['best_acc'] = float(best_acc)
    results['best_epoch'] = best_epoch
    results['optimal_temperature'] = float(optimal_temp)
    
    results_path = os.path.join(args.save_dir, f"{exp_name}_results.json")
    with open(results_path, 'w') as f:
       json.dump(results, f, indent=4)
    
    print(f"Training completed. Best test accuracy: {best_acc:.2f}% at epoch {best_epoch+1}")
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
   main()