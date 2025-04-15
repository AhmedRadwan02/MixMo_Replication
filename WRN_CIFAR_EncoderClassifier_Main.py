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

# Import models module but we'll create new classes for the experiments
from models import Wide_ResNet_Encoder, cut_mixmo, linear_mixmo

def parse_args():
    parser = argparse.ArgumentParser(description='Train MixMo models for Table 3')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_repetitions', type=int, default=2, help='Batch repetition factor (b parameter)')
    parser.add_argument('--epochs', type=int, default=300, help='Total number of epochs')
    parser.add_argument('--alpha', type=float, default=2.0, help='Alpha parameter for Beta distribution')
    parser.add_argument('--r_param', type=float, default=3.0, help='Reweighting parameter r')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

# Define model variants for each configuration in Table 3

# 1. Standard model (1 encoder, 1 classifier)
class WideResNet_1E1C(nn.Module):
    def __init__(self, depth, width, dropout_rate, num_classes):
        super(WideResNet_1E1C, self).__init__()
        print(f'| Wide-ResNet {depth}x{width} with 1 encoder, 1 classifier')
        
        # Single encoder
        self.encoder = Wide_ResNet_Encoder(depth, width, dropout_rate)
        self.feature_dim = 64 * width  # This is the channel dimension after layer3
        
        # Single classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x1, x2=None):
        # Only use x1, ignore x2
        features = self.encoder.get_embedding(x1)
        out = self.classifier(features)
        
        # Return standard model output and placeholder Nones for compatibility
        return out, None, None, None, None

# 2. Two encoders, one classifier with linear interpolation
class WideResNet_2E1C(nn.Module):
    def __init__(self, depth, width, dropout_rate, num_classes):
        super(WideResNet_2E1C, self).__init__()
        print(f'| Wide-ResNet {depth}x{width} with 2 encoders, 1 classifier')
        
        # Two encoders
        self.encoder1 = Wide_ResNet_Encoder(depth, width, dropout_rate)
        self.encoder2 = Wide_ResNet_Encoder(depth, width, dropout_rate)
        self.feature_dim = 64 * width
        
        # One classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x1, x2):
        # Extract features using both encoders
        features1 = self.encoder1.get_embedding(x1)
        features2 = self.encoder2.get_embedding(x2)
        
        # Linear mixing (as mentioned in the paper for 2E1C)
        mixed_features, kappa = linear_mixmo(features1, features2, alpha=2.0)
        
        # Use the mixed features for a single classification
        out_mix = self.classifier(mixed_features)
        
        # Return appropriate outputs for compatibility
        # Both outputs are the same as we have only one classifier
        return out_mix, out_mix, out_mix, out_mix, kappa

# 3. One encoder, two classifiers with random assignment (variant 🝓)
class WideResNet_1E2C_Random(nn.Module):
    def __init__(self, depth, width, dropout_rate, num_classes):
        super(WideResNet_1E2C_Random, self).__init__()
        print(f'| Wide-ResNet {depth}x{width} with 1 encoder, 2 classifiers (random assignment)')
        
        # Single encoder
        self.encoder = Wide_ResNet_Encoder(depth, width, dropout_rate)
        self.feature_dim = 64 * width
        
        # Two classifiers
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        self.classifier2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x1, x2):
        # Use the same encoder for both inputs
        features1 = self.encoder.get_embedding(x1)
        features2 = self.encoder.get_embedding(x2)
        
        # Randomly assign the classifiers
        out1 = self.classifier1(features1)
        out2 = self.classifier2(features2)
        
        # For mixed outputs, use a random kappa value
        kappa = torch.tensor(0.5).cuda() if torch.cuda.is_available() else torch.tensor(0.5)
        
        # Return outputs (but mixed outputs won't be used in this case)
        return out1, out2, out1, out2, kappa

# 4. One encoder, two classifiers with predominant input targeting (variant ⊗)
class WideResNet_1E2C_Targeted(nn.Module):
    def __init__(self, depth, width, dropout_rate, num_classes):
        super(WideResNet_1E2C_Targeted, self).__init__()
        print(f'| Wide-ResNet {depth}x{width} with 1 encoder, 2 classifiers (targeted assignment)')
        
        # Single encoder
        self.encoder = Wide_ResNet_Encoder(depth, width, dropout_rate)
        self.feature_dim = 64 * width
        
        # Two classifiers
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        self.classifier2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x1, x2):
        # Use the same encoder for both inputs
        features1 = self.encoder.get_embedding(x1)
        features2 = self.encoder.get_embedding(x2)
        
        # Mix features with linear mixing
        mixed_features, kappa = linear_mixmo(features1, features2, alpha=2.0)
        
        # In the ⊗ version, first classifier targets predominant input
        # The second classifier targets the other input
        out1 = self.classifier1(features1)
        out2 = self.classifier2(features2)
        
        # For mixed outputs, first classifier should predict y1 if kappa > 0.5 else y2
        # Second classifier the opposite
        out_mix1 = self.classifier1(mixed_features)
        out_mix2 = self.classifier2(mixed_features)
        
        return out1, out2, out_mix1, out_mix2, kappa

# 5. Standard Cut-MixMo (2 encoders, 2 classifiers)
class WideResNet_2E2C(nn.Module):
    def __init__(self, depth, width, dropout_rate, num_classes):
        super(WideResNet_2E2C, self).__init__()
        print(f'| Wide-ResNet {depth}x{width} with 2 encoders, 2 classifiers (Cut-MixMo)')
        
        # Two encoders
        self.encoder1 = Wide_ResNet_Encoder(depth, width, dropout_rate)
        self.encoder2 = Wide_ResNet_Encoder(depth, width, dropout_rate)
        self.feature_dim = 64 * width
        
        # Two classifiers
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        self.classifier2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        self.mixing_mode = 'patch'  # Default to patch mixing
    
    def set_mixing_mode(self, mode):
        self.mixing_mode = mode
    
    def forward(self, x1, x2):
        # Extract features using the two encoders
        features1 = self.encoder1.get_embedding(x1)
        features2 = self.encoder2.get_embedding(x2)
        
        # Original outputs (without mixing)
        out1 = self.classifier1(features1)
        out2 = self.classifier2(features2)
        
        # Use patch mixing or linear mixing based on mode
        if self.mixing_mode == 'patch':
            mixed_features, kappa = cut_mixmo(features1, features2, alpha=2.0)
        else:
            mixed_features, kappa = linear_mixmo(features1, features2, alpha=2.0)
        
        # Get mixed outputs
        out_mix1 = self.classifier1(mixed_features)
        out_mix2 = self.classifier2(mixed_features)
        
        return out1, out2, out_mix1, out_mix2, kappa

# Function to get model based on configuration
def get_model(config, args, num_classes):
    if config == '1e1c':
        return WideResNet_1E1C(28, args.width, 0.3, num_classes)
    elif config == '2e1c':
        return WideResNet_2E1C(28, args.width, 0.3, num_classes)
    elif config == '1e2c_random':
        return WideResNet_1E2C_Random(28, args.width, 0.3, num_classes)
    elif config == '1e2c_targeted':
        return WideResNet_1E2C_Targeted(28, args.width, 0.3, num_classes)
    elif config == '2e2c':
        return WideResNet_2E2C(28, args.width, 0.3, num_classes)
    else:
        raise ValueError(f"Unknown configuration: {config}")

# Modified loss function to handle different model configurations
def mixmo_loss(outputs, targets, model_config, r=3.0):
    out1, out2, out_mix1, out_mix2, kappa = outputs
    y1, y2 = targets
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    if model_config == '1e1c':
        # Just a single classification loss
        return criterion(out1, y1).mean()
    
    elif model_config == '2e1c':
        # Single classifier with mixed targets
        loss = criterion(out1, y1) * kappa + criterion(out1, y2) * (1-kappa)
        return loss.mean()
    
    elif model_config == '1e2c_random':
        # Two separate classification losses
        loss1 = criterion(out1, y1).mean()
        loss2 = criterion(out2, y2).mean()
        return loss1 + loss2
    
    elif model_config in ['1e2c_targeted', '2e2c']:
        # Standard MixMo loss with reweighting
        loss1 = criterion(out_mix1, y1)
        loss2 = criterion(out_mix2, y2)
        
        if kappa is not None:
            weight1 = 2 * (kappa ** (1 / r)) / ((kappa ** (1 / r)) + ((1 - kappa) ** (1 / r)))
            weight2 = 2 * ((1 - kappa) ** (1 / r)) / ((kappa ** (1 / r)) + ((1 - kappa) ** (1 / r)))
            loss1 = (loss1 * weight1).mean()
            loss2 = (loss2 * weight2).mean()
        else:
            loss1 = loss1.mean()
            loss2 = loss2.mean()
        
        return loss1 + loss2
    
    else:
        raise ValueError(f"Unknown model configuration: {model_config}")

def compute_dre(pred1, pred2, true_targets):
    err1 = pred1.ne(true_targets)
    err2 = pred2.ne(true_targets)
    different_errors = (err1 & err2 & pred1.ne(pred2)).sum().item()
    simultaneous_errors = (err1 & err2).sum().item()
    return (different_errors / simultaneous_errors) if simultaneous_errors > 0 else 0.0

def evaluate(model, test_loader, device, model_config):
    model.eval()
    correct_ensemble = 0
    correct1 = 0
    correct2 = 0
    total = 0
    dre_sum = 0.0
    batch_count = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if model_config == '1e1c':
                # For standard model, use same input twice
                out1, _, _, _, _ = model(inputs, inputs)
                # Single output is the ensemble output
                ensemble_output = out1
                predicted_ensemble = ensemble_output.max(1)[1]
                predicted1 = predicted_ensemble
                total_loss += criterion(out1, targets).item() * inputs.size(0)
                
                # For 1e1c, ensemble accuracy equals individual accuracy
                correct_ensemble += predicted_ensemble.eq(targets).sum().item()
                correct1 += predicted1.eq(targets).sum().item()
                correct2 = correct1  # Same as correct1
                
            else:
                # For all other models, pass the same input twice
                out1, out2, _, _, _ = model(inputs, inputs)
                
                # For models with two outputs
                if out2 is not None:
                    ensemble_output = (out1 + out2) / 2
                    _, predicted_ensemble = ensemble_output.max(1)
                    _, predicted1 = out1.max(1)
                    _, predicted2 = out2.max(1)
                    
                    correct_ensemble += predicted_ensemble.eq(targets).sum().item()
                    correct1 += predicted1.eq(targets).sum().item()
                    correct2 += predicted2.eq(targets).sum().item()
                    
                    # Only compute DRE for models with two distinct outputs
                    dre_sum += compute_dre(predicted1, predicted2, targets)
                else:
                    # For models with just one output
                    _, predicted1 = out1.max(1)
                    correct_ensemble += predicted1.eq(targets).sum().item()
                    correct1 += predicted1.eq(targets).sum().item()
                    correct2 = correct1
                
                # Compute loss
                total_loss += criterion(out1, targets).item() * inputs.size(0)
                if out2 is not None:
                    total_loss += criterion(out2, targets).item() * inputs.size(0)
            
            batch_count += 1
            total += targets.size(0)

    acc_ensemble = 100. * correct_ensemble / total
    
    if model_config in ['1e1c', '2e1c']:
        # For models with 1 classifier, individual accuracy equals ensemble
        acc_individual = acc_ensemble
    else:
        # For models with 2 classifiers, average individual accuracies
        acc_individual = (100. * correct1 / total + 100. * correct2 / total) / 2
    
    avg_loss = total_loss / total
    
    if model_config in ['1e1c', '2e1c', '1e2c_random']:
        # These models don't have meaningful diversity
        diversity = 0.0
    else:
        diversity = dre_sum / batch_count if batch_count > 0 else 0.0
        
    return acc_ensemble, acc_individual, diversity, avg_loss

def train_and_average_metrics(args, model, train_loader, test_loader, device, model_config):
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
    history = []

    for epoch in range(args.epochs):
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch_size = inputs.size(0) // 2
            x1, x2 = torch.chunk(inputs, 2)
            y1, y2 = torch.chunk(targets, 2)
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)

            # Set mixing mode to patch for Cut-MixMo model
            if model_config == '2e2c' and hasattr(model, 'set_mixing_mode'):
                use_patch_mixing = torch.rand(1).item() < 0.5  # p=0.5
                model.set_mixing_mode('patch' if use_patch_mixing else 'linear')

            # Forward pass
            outputs = model(x1, x2)
            
            # Compute loss based on model configuration
            loss = mixmo_loss(outputs, (y1, y2), model_config, r=args.r_param)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Collect metrics only for the last 10 epochs
        if epoch >= args.epochs - 10:
            acc_ensemble, acc_individual, diversity, test_loss = evaluate(model, test_loader, device, model_config)
            history.append((acc_ensemble, acc_individual, diversity, test_loss))

    # Average metrics over the last 10 epochs
    avg_metrics = np.mean(np.array(history), axis=0)
    return avg_metrics

def compute_nllc(model, test_loader, val_loader, device, model_config):
    """Compute calibrated NLL (NLLc) using temperature scaling"""
    # First, determine the optimal temperature on validation set
    model.eval()
    
    if model_config == '1e1c':
        # For single output models
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                out1, _, _, _, _ = model(inputs, inputs)
                logits_list.append(out1)
                labels_list.append(targets)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Search for optimal temperature
        best_temp = 1.0
        best_nll = float('inf')
        
        for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
            scaled_logits = logits / temp
            loss = nn.CrossEntropyLoss()(scaled_logits, labels)
            if loss.item() < best_nll:
                best_nll = loss.item()
                best_temp = temp
        
        # Now compute NLLc on test set
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                out1, _, _, _, _ = model(inputs, inputs)
                logits_list.append(out1)
                labels_list.append(targets)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Apply optimal temperature
        scaled_logits = logits / best_temp
        nllc = nn.CrossEntropyLoss()(scaled_logits, labels).item()
    
    else:
        # For ensemble models
        logits1_list = []
        logits2_list = []
        labels_list = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                out1, out2, _, _, _ = model(inputs, inputs)
                logits1_list.append(out1)
                if out2 is not None:
                    logits2_list.append(out2)
                labels_list.append(targets)
        
        logits1 = torch.cat(logits1_list)
        labels = torch.cat(labels_list)
        
        if logits2_list:
            logits2 = torch.cat(logits2_list)
            ensemble_logits = (logits1 + logits2) / 2
        else:
            ensemble_logits = logits1
        
        # Search for optimal temperature
        best_temp = 1.0
        best_nll = float('inf')
        
        for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
            scaled_logits = ensemble_logits / temp
            loss = nn.CrossEntropyLoss()(scaled_logits, labels)
            if loss.item() < best_nll:
                best_nll = loss.item()
                best_temp = temp
        
        # Now compute NLLc on test set
        logits1_list = []
        logits2_list = []
        labels_list = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                out1, out2, _, _, _ = model(inputs, inputs)
                logits1_list.append(out1)
                if out2 is not None:
                    logits2_list.append(out2)
                labels_list.append(targets)
        
        logits1 = torch.cat(logits1_list)
        labels = torch.cat(labels_list)
        
        if logits2_list:
            logits2 = torch.cat(logits2_list)
            ensemble_logits = (logits1 + logits2) / 2
        else:
            ensemble_logits = logits1
        
        # Apply optimal temperature
        scaled_logits = ensemble_logits / best_temp
        nllc = nn.CrossEntropyLoss()(scaled_logits, labels).item()
    
    return nllc

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10 if args.dataset == 'cifar10' else 100
    data_handler = DataHandler(data_root=args.data_root)

    if args.dataset == 'cifar10':
        train_loader, test_loader = data_handler.get_cifar10(batch_size=args.batch_size, batch_repetitions=2)
    else:
        train_loader, test_loader = data_handler.get_cifar100(batch_size=args.batch_size, batch_repetitions=2)
    
    # For temperature scaling, split test set into two halves
    test_size = len(test_loader.dataset)
    val_size = test_size // 2
    test_indices = list(range(test_size))
    np.random.shuffle(test_indices)
    val_indices = test_indices[:val_size]
    test_indices = test_indices[val_size:]
    
    val_dataset = Subset(test_loader.dataset, val_indices)
    test_dataset = Subset(test_loader.dataset, test_indices)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Configuration to test
    configs = [
        '1e1c',            # 1 encoder, 1 classifier
        '2e1c',            # 2 encoders, 1 classifier
        '1e2c_random',     # 1 encoder, 2 classifiers (random assignment)
        '1e2c_targeted',   # 1 encoder, 2 classifiers (targeted, ⊗ symbol in paper)
        '2e2c'             # 2 encoders, 2 classifiers (standard Cut-MixMo)
    ]
    
    results = []
    
    for config in configs:
        print(f"\n===== Testing configuration: {config} =====")
        
        model = get_model(config, args, num_classes).to(device)
        acc_ensemble, acc_individual, diversity, _ = train_and_average_metrics(
            args, model, train_loader, test_loader, device, config
        )
        
        # Compute NLLc
        nllc = compute_nllc(model, test_loader, val_loader, device, config)
        
        config_name = {
            '1e1c': '1 Enc., 1 Clas.',
            '2e1c': '2 Enc., 1 Clas.',
            '1e2c_random': '1 Enc., 2 Clas. 🝓',
            '1e2c_targeted': '1 Enc., 2 Clas. ⊗',
            '2e2c': '2 Enc., 2 Clas.'
        }[config]
        
        results.append({
            'config': config_name,
            'ensemble_acc': acc_ensemble,
            'individual_acc': acc_individual,
            'diversity_dre': diversity,
            'nllc': nllc
        })
        
        print(f"Results for {config_name}:")
        print(f"  Ensemble Accuracy: {acc_ensemble:.2f}%")
        print(f"  Individual Accuracy: {acc_individual:.2f}%")
        print(f"  Diversity (DRE): {diversity:.3f}")
        print(f"  NLLc: {nllc:.3f}")

    # Save results
    result_file = os.path.join(args.save_dir, f"table3_results.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {result_file}")
    
    # Print results in a tabular format like in the paper
    print("\nTable 3: Number of encoders/classifiers.\n")
    print("# Enc. # Clas. NLLc ↓")
    for result in results:
        config_parts = result['config'].split('.')
        num_enc = config_parts[0].strip()
        num_clas = config_parts[1].strip() if len(config_parts) > 1 else ""
        nllc = result['nllc']
        print(f"{num_enc} {num_clas} {nllc:.3f}")

if __name__ == '__main__':
    main()
