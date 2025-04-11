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
from models import Wide_ResNet28

def parse_args():
    parser = argparse.ArgumentParser(description='Train MixMo models for Figure 7 (mixing probability experiment)')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--approach', type=str, required=True, 
                      choices=['linear_mixmo','cut_mixmo','linear_mixmo_cutmix', 'cut_mixmo_cutmix'])
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_repetitions', type=int, default=2, help='Batch repetition factor (b parameter)')
    parser.add_argument('--epochs', type=int, default=300, help='Total number of epochs')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter for Beta distribution')
    parser.add_argument('--data_root', type=str, default='/home/pariamdz/projects/def-hinat/pariamdz/MixMo/datasets')
    parser.add_argument('--save_dir', type=str, default='/home/pariamdz/projects/def-hinat/pariamdz/MixMo/results/figure7')
    parser.add_argument('--run_number', type=int, default=1, help='Run 1, 2, or 3 for averaging')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_mix_prob', type=float, default=0.0, help='Minimum mixing probability to test')
    parser.add_argument('--max_mix_prob', type=float, default=1.0, help='Maximum mixing probability to test')
    parser.add_argument('--mix_prob_step', type=float, default=0.1, help='Step size for mixing probability')
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
    if args.approach in ['linear_mixmo','linear_mixmo_cutmix']:
        aug_type = 'LinearMixMo'
    elif args.approach in ['cut_mixmo','cut_mixmo_cutmix']:
        aug_type = 'CutMixMo'
    
    return Wide_ResNet28(
        widen_factor=args.width,
        dropout_rate=0.3,
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

def compute_dre(pred1, pred2, true_targets):
    """
    Compute Different Representation Errors (DRE) metric for diversity measurement
    
    This calculates the proportion of times where both subnets make errors, 
    but they make different errors
    """
    err1 = pred1.ne(true_targets)
    err2 = pred2.ne(true_targets)
    different_errors = (err1 & err2 & pred1.ne(pred2)).sum().item()
    simultaneous_errors = (err1 & err2).sum().item()
    return (different_errors / simultaneous_errors) if simultaneous_errors > 0 else 0.0

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

def train_epoch(model, train_loader, optimizer, device, args, epoch=0, total_epochs=None, mix_prob=0.5):
    """Unified training function for all approaches with controllable mixing probability"""
    model.train()
    running_loss = 0.0
    mixing_prob = mix_prob  # Use the provided mixing probability
    
    # Calculate when to start decreasing probability (last 1/12 of training)
    if args.approach in ['cut_mixmo','cut_mixmo_cutmix'] and total_epochs is not None:
        decay_start = total_epochs - total_epochs // 12
        
        if epoch >= decay_start:
            # Linear decay from mixing_prob to 0 over the last 1/12 epochs
            progress = (epoch - decay_start) / (total_epochs - decay_start)
            mixing_prob = mixing_prob * (1 - progress)
    
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
        if args.approach in ['cut_mixmo','cut_mixmo_cutmix']:
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
    """Evaluation function for all approaches with NLLc calculation and diversity metrics"""
    model.eval()
    
    # MixMo evaluation
    correct_ensemble = 0
    correct1 = 0
    correct2 = 0
    correct_top5 = 0
    total = 0
    all_logits = []
    all_targets = []
    
    # For DRE calculation
    total_dre = 0.0
    batch_count = 0
    
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
            
            # Calculate DRE for this batch
            dre = compute_dre(predicted1, predicted2, true_targets)
            total_dre += dre
            batch_count += 1
            
            total += targets.size(0)
    
    acc_ensemble = 100. * correct_ensemble / total
    acc_top5 = 100. * correct_top5 / total
    acc1 = 100. * correct1 / total
    acc2 = 100. * correct2 / total
    avg_individual_acc = (acc1 + acc2) / 2
    
    # Average DRE across all batches
    diversity = total_dre / batch_count if batch_count > 0 else 0.0
    
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
        
    return acc_ensemble, acc_top5, acc1, acc2, avg_individual_acc, diversity, nllc

def train_model_with_mixing_prob(args, mixing_prob, data_handler, device, num_classes):
    """Train a model with a specific mixing probability and evaluate it"""
    print(f"\n===== Training with mixing probability p = {mixing_prob:.1f} =====")
    
    # Configure data loaders based on approach
    batch_repetitions = args.batch_repetitions
    
    if args.approach in ['cut_mixmo_cutmix', 'linear_mixmo_cutmix']:
        # For CutMix approaches, we need to load the dataset first without repetitions
        # and then apply CutMix and repetitions in one step
        if args.dataset == 'cifar10':
            # Get the dataset without repetitions first (batch_repetitions=1)
            train_loader, test_loader = data_handler.get_cifar10(
                batch_size=args.batch_size, 
                batch_repetitions=1  # No repetitions yet
            )
        else:  # cifar100
            train_loader, test_loader = data_handler.get_cifar100(
                batch_size=args.batch_size, 
                batch_repetitions=1  # No repetitions yet
            )
            
        # Now apply CutMix and repetitions together
        train_loader = data_handler.get_cutmix_loader(
            dataset=train_loader.dataset,  # Use the dataset, not the loader
            batch_size=args.batch_size * batch_repetitions,
            alpha=args.alpha,
            batch_repetitions=batch_repetitions,  # Apply repetitions here
            num_classes=num_classes
        )
    else:
        # For non-CutMix approaches, use the regular loading with repetitions
        if args.dataset == 'cifar10':
            train_loader, test_loader = data_handler.get_cifar10(
                batch_size=args.batch_size * batch_repetitions, 
                batch_repetitions=batch_repetitions
            )
        else:  # cifar100
            train_loader, test_loader = data_handler.get_cifar100(
                batch_size=args.batch_size * batch_repetitions, 
                batch_repetitions=batch_repetitions
            )
    
    # Create and configure model
    model = get_model(args, num_classes).to(device)
    
    # Configure optimizer and scheduler
    initial_lr = 0.1 / args.batch_repetitions * (args.batch_size / 128)
    optimizer = optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=0.9,
        weight_decay=3e-4
    )
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 225], gamma=0.1)
    
    # Generate experiment name
    exp_name = f"{args.approach}_p{mixing_prob:.1f}_b{batch_repetitions}_wrn{args.width}_{args.dataset}_run{args.run_number}"
    print(f"Starting experiment: {exp_name}")
    
    # Initialize results storage
    results = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_top5_acc': [],
        'test_acc_subnet1': [],
        'test_acc_subnet2': [],
        'test_diversity': [],
        'test_nllc': []
    }
    
    best_acc = 0.0
    best_epoch = 0
    final_metrics = {}
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train the model with the specific mixing probability
        train_loss, train_acc, train_acc1, train_acc2 = train_epoch(
            model, train_loader, optimizer, device, args, 
            epoch=epoch, total_epochs=args.epochs, mix_prob=mixing_prob
        )
        scheduler.step()
        
        # Regular evaluation for training monitoring (without calculating NLLc)
        test_acc, test_top5_acc, test_acc1, test_acc2, avg_individual_acc, diversity, _ = evaluate(
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
        results['test_diversity'].append(float(diversity))
        results['test_nllc'].append(None)  # Will calculate NLLc at the end
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Acc: {test_acc:.2f}%, Test Top-5 Acc: {test_top5_acc:.2f}%")
        print(f"Test Acc Subnet1: {test_acc1:.2f}%, Subnet2: {test_acc2:.2f}%")
        print(f"Diversity (DRE): {diversity:.4f}")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            
            # Save model
            model_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
            
        # For Figure 7, we want to collect final results, especially from the last 10 epochs
        if epoch >= args.epochs - 10:
            # Store metrics for later averaging
            if 'final_metrics' not in final_metrics:
                final_metrics['final_metrics'] = []
            
            final_metrics['final_metrics'].append({
                'ensemble_acc': float(test_acc),
                'individual_acc': float(avg_individual_acc),
                'diversity': float(diversity)
            })
    
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
    test_acc, test_top5_acc, test_acc1, test_acc2, avg_individual_acc, diversity, nllc = evaluate(
        model, eval_loader, device, args,
        optimal_temp=optimal_temp
    )
    
    # Update results with final values
    results['final_acc_ensemble'] = float(test_acc)
    results['final_acc_top5'] = float(test_top5_acc)
    results['final_acc_subnet1'] = float(test_acc1)
    results['final_acc_subnet2'] = float(test_acc2)
    results['final_avg_individual_acc'] = float(avg_individual_acc)
    results['final_diversity'] = float(diversity)
    results['final_nllc'] = float(nllc) if nllc is not None else None
    results['best_acc'] = float(best_acc)
    results['best_epoch'] = best_epoch
    results['optimal_temperature'] = float(optimal_temp)
    
    # Save detailed results for this mixing probability
    results_path = os.path.join(args.save_dir, f"{exp_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Training completed. Best test accuracy: {best_acc:.2f}% at epoch {best_epoch+1}")
    print(f"Final metrics - Ensemble acc: {test_acc:.2f}%, Individual acc: {avg_individual_acc:.2f}%, Diversity: {diversity:.4f}")
    
    # Calculate average metrics from the last 10 epochs
    avg_metrics = {
        'p': mixing_prob,
        'ensemble_acc': np.mean([m['ensemble_acc'] for m in final_metrics['final_metrics']]),
        'individual_acc': np.mean([m['individual_acc'] for m in final_metrics['final_metrics']]),
        'diversity_dre': np.mean([m['diversity'] for m in final_metrics['final_metrics']])
    }
    
    return avg_metrics

def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataset specific parameters
    num_classes = 10 if args.dataset == 'cifar10' else 100
    
    # Setup data handler
    data_handler = DataHandler(data_root=args.data_root)
    
    # Figure 7 experiment: vary mixing probability and measure performance
    mix_probs = np.arange(args.min_mix_prob, args.max_mix_prob + args.mix_prob_step, args.mix_prob_step)
    figure7_results = []
    
    for mix_prob in mix_probs:
        # Set seed based on run number and reset for each mixing probability
        seed = args.seed + args.run_number - 1
        set_seed(seed)
        
        # Train and evaluate model with this mixing probability
        avg_metrics = train_model_with_mixing_prob(
            args, mix_prob, data_handler, device, num_classes
        )
        
        figure7_results.append(avg_metrics)
    
    # Save the combined results for Figure 7
    figure7_path = os.path.join(args.save_dir, f"figure7_results_{args.approach}_run{args.run_number}.json")
    with open(figure7_path, 'w') as f:
        json.dump(figure7_results, f, indent=4)
    
    print(f"Figure 7 experiment completed. Results saved to {figure7_path}")
    
    # Print summary of results
    print("\nSummary of results for different mixing probabilities:")
    print("p\tEnsemble Acc\tIndividual Acc\tDiversity")
    for result in figure7_results:
        print(f"{result['p']:.1f}\t{result['ensemble_acc']:.2f}%\t{result['individual_acc']:.2f}%\t{result['diversity_dre']:.4f}")

if __name__ == "__main__":
    main()
