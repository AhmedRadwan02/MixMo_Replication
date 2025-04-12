import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import json
from tqdm import tqdm
from data_handler import DataHandler
from models import Wide_ResNet28

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate models on CIFAR-100-C')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model checkpoint')
    parser.add_argument('--approach', type=str, required=True, 
                      choices=['linear_mixmo', 'cut_mixmo', 'linear_mixmo_cutmix', 'cut_mixmo_cutmix'])
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--severity', type=int, default=3,
                      help='Corruption severity level (1-5)')
    parser.add_argument('--augmix_trained', action='store_true',
                      help='Flag if model was trained with AugMix')
    parser.add_argument('--save_dir', type=str, default='./results',
                      help='Directory to save results')
    return parser.parse_args()

def get_model(args, num_classes=100):
    """Create model based on specified approach"""
    aug_type = 'none'
    if args.approach in ['linear_mixmo', 'linear_mixmo_cutmix']:
        aug_type = 'LinearMixMo'
    elif args.approach in ['cut_mixmo', 'cut_mixmo_cutmix']:
        aug_type = 'CutMixMo'
    
    return Wide_ResNet28(
        widen_factor=args.width,
        dropout_rate=0.3,
        num_classes=num_classes,
        augmentation_type=aug_type,
    )

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

def evaluate(model, test_loader, device):
    """Evaluation function for corrupted datasets exactly matching the paper"""
    model.eval()
    
    correct_ensemble = 0
    correct_top5 = 0
    total = 0
    
    # Using sum of losses as in the paper
    criterion = nn.CrossEntropyLoss(reduction='sum')
    all_losses = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # For inference, use the same input for both branches as stated in the paper:
            # "At inference, the same input x is repeated twice"
            out1, out2, _, _, _ = model(inputs, inputs)
            
            # Ensemble prediction (average of both outputs):
            # "Then, the diverse predictions are averaged: 1/2(ŷ0 + ŷ1)"
            ensemble_output = (out1 + out2) / 2
            
            # Calculate raw loss (NLL) directly with sum reduction
            loss = criterion(ensemble_output, targets).item()
            all_losses.append(loss)
            
            # Top-1 accuracy
            _, predicted = ensemble_output.max(1)
            correct_ensemble += predicted.eq(targets).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = ensemble_output.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            top5_correct = top5_pred.eq(targets.view(1, -1).expand_as(top5_pred))
            correct_top5 += top5_correct.sum().item()
            
            total += targets.size(0)
    
    # Calculate accuracy metrics
    acc_top1 = 100. * correct_ensemble / total
    acc_top5 = 100. * correct_top5 / total
    
    # Calculate NLL as sum of losses divided by total samples (matches paper's approach)
    nll = sum(all_losses) / total
    
    return acc_top1, acc_top5, nll

def main():
    args = parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")
    
    # List of corruptions from the paper (appendix 6.4 mentions these specific corruptions)
    corruptions = [
        'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
        'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
        'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
        'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur'
    ]
    
    # Create model and load weights
    model = get_model(args)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Initialize the data handler
    data_handler = DataHandler(data_root=args.data_root)
    
    # Results storage (using a dictionary for JSON output)
    results = {
        'model_info': {
            'model_path': args.model_path,
            'approach': args.approach,
            'width': args.width,
            'augmix_trained': args.augmix_trained,
            'severity': args.severity
        },
        'corruption_results': {}
    }
    
    # Track averages for summary
    avg_metrics = {
        'top1': [],
        'top5': [],
        'nll': []
    }
    
    # Evaluate on each corruption type
    for corruption in corruptions:
        print(f"\nEvaluating on corruption: {corruption}")
        
        # Load CIFAR-100-C with specific corruption
        # Severity is set to 3 as mentioned in section 6.4 of the paper
        cifar100c_loader = data_handler.get_cifar100c(
            corruption_type=corruption,
            batch_size=args.batch_size,
            severity=args.severity
        )
        
        # Evaluate model without temperature scaling - matching the paper
        top1, top5, nll = evaluate(model, cifar100c_loader, device)
        
        # Store results for this corruption
        results['corruption_results'][corruption] = {
            'top1': float(top1),
            'top5': float(top5),
            'nll': float(nll)
        }
        
        # Track for averages
        avg_metrics['top1'].append(top1)
        avg_metrics['top5'].append(top5)
        avg_metrics['nll'].append(nll)
        
        print(f"Top-1 Accuracy: {top1:.2f}%, Top-5 Accuracy: {top5:.2f}%, NLL: {nll:.4f}")
    
    # Calculate average results across all corruptions
    # This matches how they report the overall performance in Table 4
    results['average'] = {
        'top1': float(np.mean(avg_metrics['top1'])),
        'top5': float(np.mean(avg_metrics['top5'])),
        'nll': float(np.mean(avg_metrics['nll']))
    }
    
    # Print summary
    print("\n" + "="*50)
    print(f"Model: {args.model_path}")
    print(f"Approach: {args.approach}, Width: {args.width}")
    print(f"Average Top-1 Accuracy: {results['average']['top1']:.2f}%")
    print(f"Average Top-5 Accuracy: {results['average']['top5']:.2f}%")
    print(f"Average NLL: {results['average']['nll']:.4f}")
    print("="*50)
    
    # Generate filename for results
    model_identifier = f"{args.approach}_w{args.width}"
    if args.augmix_trained:
        model_identifier += "_augmix"
    
    result_filename = os.path.join(args.save_dir, f"{model_identifier}_cifar100c_results.json")
    
    # Save results as JSON
    with open(result_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {result_filename}")

if __name__ == "__main__":
    main()