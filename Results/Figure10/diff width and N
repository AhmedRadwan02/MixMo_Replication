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
    parser = argparse.ArgumentParser(description='Train MixMo models with ensemble & param counting (Figure 10 style)')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--approach', type=str, required=True, 
                      choices=['linear_mixmo','cut_mixmo','linear_mixmo_cutmix', 'cut_mixmo_cutmix'])
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--num_ensemble', type=int, default=1, 
                       help='Number of models to ensemble (N)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_repetitions', type=int, default=4, help='Batch repetition factor (b parameter)') # Set to 4 for Fig 10
    parser.add_argument('--epochs', type=int, default=None, help='If None, will use approach-specific default')
    parser.add_argument('--alpha', type=float, default=2.0, help='Alpha parameter for Beta distribution') # Changed to 2.0 as in paper
    parser.add_argument('--data_root', type=str, default='/home/pariamdz/projects/def-hinat/pariamdz/MixMo/datasets')
    parser.add_argument('--save_dir', type=str, default='/home/pariamdz/projects/def-hinat/pariamdz/MixMo/results/figure10')
    parser.add_argument('--run_number', type=int, default=1, help='Run 1, 2, or 3 for averaging')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fig10_data', action='store_true', help='Generate data for Figure 10')
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
    if args.approach in ['cut_mixmo','cut_mixmo_cutmix'] and total_epochs is not None:
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

def evaluate_single_model(model, test_loader, device, calculate_metrics=True):
    """Evaluation function for single MixMo model with NLLc calculation"""
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
    
    return acc_ensemble, acc_top5, acc1, acc2, all_logits, all_targets

def evaluate_ensemble(models, test_loader, device, calculate_metrics=True):
    """
    Evaluate multiple MixMo models as an ensemble
    """
    for model in models:
        model.eval()
    
    correct_ensemble = 0
    correct_top5 = 0
    total = 0
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating Ensemble"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Average predictions from all models
            sum_logits = 0
            for model in models:
                out1, out2, _, _, _ = model(inputs, inputs)
                # Ensemble prediction within model (average of both outputs)
                model_ensemble = (out1 + out2) / 2
                sum_logits += model_ensemble
            
            # Average across all models
            ensemble_output = sum_logits / len(models)
            
            # Save logits and targets for NLL calculation
            if calculate_metrics:
                all_logits.append(ensemble_output.detach().cpu())
                all_targets.append(targets.detach().cpu())
            
            # Handle targets that might be one-hot encoded
            if isinstance(targets, torch.Tensor) and targets.dim() > 1 and targets.size(1) > 1:
                true_targets = targets.argmax(dim=1)
            else:
                true_targets = targets
            
            # Top-1 accuracy
            _, predicted_ensemble = ensemble_output.max(1)
            correct_ensemble += predicted_ensemble.eq(true_targets).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = ensemble_output.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            top5_correct = top5_pred.eq(true_targets.view(1, -1).expand_as(top5_pred))
            correct_top5 += top5_correct.sum().item()
            
            total += targets.size(0)
    
    acc_ensemble = 100. * correct_ensemble / total
    acc_top5 = 100. * correct_top5 / total
    
    return acc_ensemble, acc_top5, all_logits, all_targets

def optimize_temperature(logits, targets, device):
    """
    Find the optimal temperature for calibration
    """
    # Set up temperature scaling model
    temperature_model = TemperatureScaling().to(device)
    
    # Set up NLL loss
    nll_criterion = nn.CrossEntropyLoss()
    
    # Optimizer for temperature parameter
    optimizer = optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)
    
    # Define the optimization step
    def eval():
        optimizer.zero_grad()
        scaled_logits = temperature_model(logits)
        loss = nll_criterion(scaled_logits, targets)
        loss.backward()
        return loss
    
    # Optimize temperature
    optimizer.step(eval)
    
    return temperature_model.temperature.item()

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    num_classes = 10 if args.dataset == 'cifar10' else 100
    
    args.epochs = 300 if args.epochs is None else args.epochs
    
    weight_decay = 3e-4 
    
    # Setup data handler
    data_handler = DataHandler(data_root=args.data_root)
    
    # Configure data loaders based on approach
    batch_repetitions = args.batch_repetitions
    print(f"Using batch repetition: b = {batch_repetitions}")
        
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
    
    # For Figure 10, we'll use an ensemble of N models
    N = args.num_ensemble
    ensemble_models = []
    
    # Track parameters for Figure 10
    single_params = None
    
    # Train or load each model in the ensemble
    for i in range(N):
        # Set different seed for each model for diversity
        model_seed = seed + i*100
        set_seed(model_seed)
        
        # Create model
        model = get_model(args, num_classes).to(device)
        
        # Count parameters (only need to do this once)
        if i == 0:
            single_params = count_parameters(model)
            print(f"Single model parameters: {single_params:,d} (~{single_params/1e6:.2f}M)")
        
        # Create optimizer
        initial_lr = 0.1 / args.batch_repetitions * (args.batch_size / 128)
        optimizer = optim.SGD(
            model.parameters(),
            lr=initial_lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 225], gamma=0.1)
        
        # Generate experiment name for this model
        exp_name = f"{args.approach}_b{batch_repetitions}_wrn{args.width}_{args.dataset}_model{i+1}_run{args.run_number}"
        print(f"Training model {i+1}/{N}: {exp_name}")
        
        # Load model if it exists
        model_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            model.load_state_dict(torch.load(model_path))
        else:
            print(f"Training new model: {exp_name}")
            
            # Initialize results storage
            model_results = {
                'epoch': [],
                'train_loss': [],
                'train_acc': [],
                'test_acc': [],
                'test_top5_acc': [],
                'test_acc_subnet1': [],
                'test_acc_subnet2': [],
            }
            
            best_acc = 0.0
            best_epoch = 0
            
            # Training loop
            for epoch in range(args.epochs):
                print(f"\nEpoch {epoch+1}/{args.epochs} for model {i+1}/{N}")
                
                # Train the model
                train_loss, train_acc, train_acc1, train_acc2 = train_epoch(
                    model, train_loader, optimizer, device, args, 
                    epoch=epoch, total_epochs=args.epochs
                )
                scheduler.step()
                
                # Evaluate model
                test_acc, test_top5_acc, test_acc1, test_acc2, _, _ = evaluate_single_model(
                    model, test_loader, device, calculate_metrics=False
                )
                
                # Log results
                model_results['epoch'].append(epoch)
                model_results['train_loss'].append(float(train_loss))
                model_results['train_acc'].append(float(train_acc))
                model_results['test_acc'].append(float(test_acc))
                model_results['test_top5_acc'].append(float(test_top5_acc))
                model_results['test_acc_subnet1'].append(float(test_acc1))
                model_results['test_acc_subnet2'].append(float(test_acc2))
                
                # Print epoch results
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Test Acc: {test_acc:.2f}%, Test Top-5 Acc: {test_top5_acc:.2f}%")
                print(f"Test Acc Subnet1: {test_acc1:.2f}%, Subnet2: {test_acc2:.2f}%")
                
                # Save best model
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch
                    
                    # Save model
                    torch.save(model.state_dict(), model_path)
                    print(f"New best model saved with accuracy: {best_acc:.2f}%")
            
            # Save training results
            model_results['best_acc'] = float(best_acc)
            model_results['best_epoch'] = best_epoch
            
            model_results_path = os.path.join(args.save_dir, f"{exp_name}_training_results.json")
            with open(model_results_path, 'w') as f:
                json.dump(model_results, f, indent=4)
            
            # Load best checkpoint for ensemble
            model.load_state_dict(torch.load(model_path))
        
        # Add model to ensemble
        model.eval()
        ensemble_models.append(model)
    
    # Calculate total parameters 
    total_params = N * single_params
    print(f"\nEnsemble info: N={N}, Width={args.width}, Total params={total_params:,d} (~{total_params/1e6:.2f}M)")
    
    # Split test set for temperature calibration
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
    
    # Evaluate ensemble on calibration set
    print("Evaluating ensemble on calibration set...")
    calib_acc, calib_top5, calib_logits_list, calib_targets_list = evaluate_ensemble(
        ensemble_models, calib_loader, device, calculate_metrics=True
    )
    
    # Prepare logits and targets for temperature scaling
    calib_logits = torch.cat(calib_logits_list, dim=0).to(device)
    calib_targets = torch.cat(calib_targets_list, dim=0).to(device)
    
    # Optimize temperature for calibration
    print("Optimizing temperature for calibration...")
    optimal_temp = optimize_temperature(calib_logits, calib_targets, device)
    print(f"Optimal temperature: {optimal_temp:.4f}")
    
    # Evaluate ensemble on evaluation set
    print("Evaluating ensemble on evaluation set...")
    eval_acc, eval_top5, eval_logits_list, eval_targets_list = evaluate_ensemble(
        ensemble_models, eval_loader, device, calculate_metrics=True
    )
    
    # Convert logits and targets to tensors
    eval_logits = torch.cat(eval_logits_list, dim=0).to(device)
    eval_targets = torch.cat(eval_targets_list, dim=0).to(device)
    
    # Calculate NLL before and after temperature scaling
    raw_nll = calculate_nll(eval_logits, eval_targets)
    scaled_logits = eval_logits / optimal_temp
    calibrated_nll = calculate_nll(scaled_logits, eval_targets)
    
    print("\n============================================")
    print("ENSEMBLE RESULTS")
    print(f"Ensemble size (N): {N}")
    print(f"Model width (w): {args.width}")
    print(f"Approach: {args.approach}")
    print(f"Final Acc (top-1): {eval_acc:.2f}%")
    print(f"Final Top-5 Acc: {eval_top5:.2f}%")
    print(f"Raw NLL: {raw_nll:.4f}")
    print(f"Calibrated NLL (NLLc): {calibrated_nll:.4f}")
    print(f"Total Parameters: {total_params:,d} (~{total_params/1e6:.2f}M)")
    print("============================================\n")
    
    # Save ensemble results
    ensemble_results = {
        'ensemble_size': N,
        'width': args.width,
        'approach': args.approach,
        'dataset': args.dataset,
        'batch_repetitions': batch_repetitions,
        'params_per_model': single_params,
        'total_params': total_params,
        'final_acc': eval_acc,
        'final_top5': eval_top5,
        'raw_nll': raw_nll,
        'calibrated_nll': calibrated_nll,
        'optimal_temp': optimal_temp
    }
    
    # Save ensemble results
    ensemble_name = f"{args.approach}_b{batch_repetitions}_wrn{args.width}_N{N}_{args.dataset}_run{args.run_number}"
    ensemble_results_path = os.path.join(args.save_dir, f"{ensemble_name}_ensemble_results.json")
    with open(ensemble_results_path, 'w') as f:
        json.dump(ensemble_results, f, indent=4)
    
    # Save Figure 10 data if requested
    if args.fig10_data:
        fig10_data = {
            'approach': args.approach,
            'width': args.width,
            'ensemble_size': N,
            'params_per_model': single_params,
            'total_params': total_params,
            'nllc': calibrated_nll,
            'nllc_per_params': calibrated_nll / (total_params / 1e6)  # NLLc per million parameters
        }
        
        fig10_path = os.path.join(args.save_dir, f"fig10_data_w{args.width}_N{N}_{args.approach}.json")
        with open(fig10_path, 'w') as f:
            json.dump(fig10_data, f, indent=4)
        print(f"Figure 10 data saved to {fig10_path}")
    
    print(f"Ensemble results saved to {ensemble_results_path}")

if __name__ == "__main__":
    main()
