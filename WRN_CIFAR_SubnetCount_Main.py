import os
import math
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
    parser = argparse.ArgumentParser(description='Train MixMo models for Figure 9 replication')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--approach', type=str, required=True, 
                        choices=['linear_mixmo','cut_mixmo','linear_mixmo_cutmix', 'cut_mixmo_cutmix'])
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_repetitions', type=int, default=2, help='Batch repetition factor (b parameter)')
    parser.add_argument('--epochs', type=int, default=300, help='Total number of epochs')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter for Beta/Dirichlet distribution')
    parser.add_argument('--mix_prob', type=float, default=0.5, help='Mixing probability p for patch vs. linear')
    parser.add_argument('--r_param', type=float, default=3.0, help='Reweighting parameter r')
    parser.add_argument('--data_root', type=str, default='/home/pariamdz/projects/def-hinat/pariamdz/MixMo/datasets')
    parser.add_argument('--save_dir', type=str, default='/home/pariamdz/projects/def-hinat/pariamdz/MixMo/results/figure9')
    parser.add_argument('--run_number', type=int, default=1, help='Run 1, 2, or 3 for averaging')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--M_values', type=str, default='2,3,4,5', help='Comma-separated list of M values to test')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

# Multi-network model for M > 2 experiments
class Wide_ResNet_Multi(nn.Module):
    """
    WideResNet with multiple subnetworks for M >= 2.
    Each subnetwork (an instance of Wide_ResNet28) produces 2 outputs;
    therefore, we create ceil(M/2) subnetworks. For M==2 the base model is used.
    """
    def __init__(self, depth, width, dropout_rate, num_classes, num_networks, augmentation_type):
        super(Wide_ResNet_Multi, self).__init__()
        self.num_networks = num_networks
        num_subnets = math.ceil(num_networks / 2)
        self.subnetworks = nn.ModuleList([
            Wide_ResNet28(
                widen_factor=width,
                dropout_rate=dropout_rate,
                num_classes=num_classes,
                augmentation_type=augmentation_type
            ) for _ in range(num_subnets)
        ])
    
    def set_mixing_mode(self, mode):
        for subnet in self.subnetworks:
            if hasattr(subnet, 'set_mixing_mode'):
                subnet.set_mixing_mode(mode)
    
    def forward(self, x, x2=None):
        """
        For M==2, two inputs are passed (x and x2).
        For M > 2, only a single input is provided (which is duplicated internally).
        The outputs (each subnetwork returns two outputs) are concatenated
        and truncated to exactly num_networks outputs.
        """
        outputs = []
        if x2 is not None:
            for subnet in self.subnetworks:
                out1, out2, out_mix1, out_mix2, kappa = subnet(x, x2)
                outputs.extend([out1, out2])
        else:
            for subnet in self.subnetworks:
                out1, out2, out_mix1, out_mix2, kappa = subnet(x, x)
                outputs.extend([out1, out2])
        
        if len(outputs) > self.num_networks:
            outputs = outputs[:self.num_networks]
        
        return outputs

def get_model(args, num_classes, M_value):
    """
    Create the model based on the specified approach and M.
    For M==2, the base Wide_ResNet28 is returned.
    For M > 2, a multi-network model is built.
    The augmentation type is chosen based on the approach.
    """
    if args.approach in ['linear_mixmo', 'linear_mixmo_cutmix']:
        aug_type = 'LinearMixMo'
    elif args.approach in ['cut_mixmo', 'cut_mixmo_cutmix']:
        aug_type = 'CutMixMo'
    
    if M_value == 2:
        return Wide_ResNet28(
            widen_factor=args.width,
            dropout_rate=0.3,
            num_classes=num_classes,
            augmentation_type=aug_type,
        )
    else:
        return Wide_ResNet_Multi(
            depth=28,
            width=args.width,
            dropout_rate=0.3,
            num_classes=num_classes,
            num_networks=M_value,
            augmentation_type=aug_type,
        )

def sample_mixing_ratios(M, alpha, device):
    """
    Sample mixing ratios from a Dirichlet distribution.
    For M=2, equivalent to Beta(alpha, alpha).
    For M>2, uses general Dirichlet distribution.
    """
    if M == 2:
        # Use Beta distribution (special case of Dirichlet for M=2)
        beta_dist = torch.distributions.Beta(alpha, alpha)
        kappa = beta_dist.sample().to(device)
        return torch.tensor([kappa, 1 - kappa], device=device)
    else:
        # Use Dirichlet distribution for M > 2
        concentration = torch.ones(M, device=device) * alpha
        dirichlet = torch.distributions.Dirichlet(concentration)
        return dirichlet.sample()

def generalized_weighting(kappa_values, r=3.0):
    """
    Generalized weighting function for M>=2 as per equation (5) in the paper.
    """
    M = len(kappa_values)
    kappa_power = kappa_values ** (1/r)
    weights = M * kappa_power / kappa_power.sum()
    return weights

def mixmo_loss(outputs, targets, kappa=None, r=3.0):
    """
    MixMo loss weighting scheme for the two-branch (M==2) case.
    """
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

def mixmo_loss_MN(outputs, targets, kappa_values, r=3.0):
    """
    MixMo loss weighting scheme for M > 2 case.
    """
    criterion = nn.CrossEntropyLoss(reduction='none')
    M = len(outputs)
    
    # Calculate the weights for each output
    weights = generalized_weighting(kappa_values, r)
    
    # Calculate weighted loss
    total_loss = 0.0
    for i in range(M):
        loss_i = criterion(outputs[i], targets[i])
        weighted_loss_i = (loss_i * weights[i]).mean()
        total_loss += weighted_loss_i
    
    return total_loss

def compute_dre(pred1, pred2, true_targets):
    """
    Compute Different Representation Errors (DRE) metric for diversity measurement
    """
    err1 = pred1.ne(true_targets)
    err2 = pred2.ne(true_targets)
    different_errors = (err1 & err2 & pred1.ne(pred2)).sum().item()
    simultaneous_errors = (err1 & err2).sum().item()
    return (different_errors / simultaneous_errors) if simultaneous_errors > 0 else 0.0

def linear_mix_MN(features_list, kappa_values):
    """
    Linear mixing of M features with mixing ratios kappa_values.
    """
    M = len(features_list)
    mixed_features = 0
    for i in range(M):
        mixed_features += kappa_values[i] * features_list[i]
    
    return M * mixed_features  # Multiply by M to preserve scale

def patch_mix_MN(features_list, kappa_values):
    """
    Patch-based mixing of M features with mixing ratios kappa_values.
    First linearly interpolates M-1 inputs, then patches a region from the M-th.
    """
    M = len(features_list)
    
    # Randomly select one feature to be patched in
    k = torch.randint(0, M, (1,)).item()
    
    # Create a binary mask for the patch
    mask = torch.zeros_like(features_list[0])
    _, _, H, W = mask.shape
    
    # Determine patch size based on the k-th ratio
    patch_ratio = kappa_values[k].item()
    patch_size = int(math.sqrt(patch_ratio) * min(H, W))
    
    # Randomly position the patch
    x = torch.randint(0, W - patch_size + 1, (1,)).item()
    y = torch.randint(0, H - patch_size + 1, (1,)).item()
    
    # Create the binary mask
    mask[:, :, y:y+patch_size, x:x+patch_size] = 1.0
    
    # Mix other features using linear mixing with adjusted ratios
    other_indices = [i for i in range(M) if i != k]
    other_ratios = kappa_values[other_indices]
    
    # Normalize other ratios to sum to 1
    other_ratios = other_ratios / other_ratios.sum()
    
    # Linearly mix the remaining features
    linear_mixed = 0
    for i, idx in enumerate(other_indices):
        linear_mixed += other_ratios[i] * features_list[idx]
    
    # Combine the patch with linear mixed features
    mixed_features = mask * features_list[k] + (1 - mask) * linear_mixed
    
    return M * mixed_features  # Multiply by M to preserve scale

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

def optimize_temperature(model, val_loader, device, args, M_value):
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
            
            # For M=2
            if M_value == 2:
                # For inference, use the same input for both branches
                out1, out2, _, _, _ = model(inputs, inputs)
                
                # Ensemble prediction (average of both outputs)
                ensemble_output = (out1 + out2) / 2
                logits_list.append(ensemble_output)
            # For M>2
            else:
                outputs = model(inputs)
                ensemble_output = torch.stack(outputs).mean(dim=0)
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

def train_epoch(model, train_loader, optimizer, device, args, M_value, epoch=0, total_epochs=None):
    """
    Unified training function for all approaches.
    For M==2 the standard MixMo loss is used; for M>2, the ensemble output
    is computed and a standard cross-entropy loss is applied.
    """
    model.train()
    running_loss = 0.0
    mixing_prob = args.mix_prob  # Fixed mixing probability
    
    # Calculate when to start decreasing probability (last 1/12 of training)
    if args.approach in ['cut_mixmo', 'cut_mixmo_cutmix'] and total_epochs is not None:
        decay_start = total_epochs - total_epochs // 12
        
        if epoch >= decay_start:
            # Linear decay from mixing_prob to 0 over the last 1/12 epochs
            progress = (epoch - decay_start) / (total_epochs - decay_start)
            mixing_prob = mixing_prob * (1 - progress)
    
    batch_repetitions = args.batch_repetitions
    
    # For M=2, use the standard approach
    if M_value == 2:
        correct1 = 0
        correct2 = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc="Training (M=2)"):
            # Reshape to get the repeated samples
            batch_size = inputs.size(0) // batch_repetitions
            x_chunks = torch.chunk(inputs, batch_repetitions)
            y_chunks = torch.chunk(targets, batch_repetitions)
            
            # Take first two chunks for MixMo
            x1, x2 = x_chunks[0].to(device), x_chunks[1].to(device)
            y1, y2 = y_chunks[0].to(device), y_chunks[1].to(device)
            
            # For Cut-MixMo, decide whether to use patch mixing or linear mixing
            use_patch_mixing = True  # Default for Cut-MixMo is patch mixing
            if args.approach in ['cut_mixmo', 'cut_mixmo_cutmix']:
                use_patch_mixing = torch.rand(1).item() < mixing_prob
            
            # Set mixing mode before forward pass
            if hasattr(model, 'set_mixing_mode'):
                model.set_mixing_mode('patch' if use_patch_mixing else 'linear')
            
            # Forward pass
            out1, out2, out_mix1, out_mix2, kappa = model(x1, x2)
            
            # Calculate loss
            loss = mixmo_loss((out1, out2, out_mix1, out_mix2), (y1, y2), kappa, r=args.r_param)
            
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
        individual_acc = [acc1, acc2]
        
        return running_loss / len(train_loader), avg_acc, individual_acc
    
    # For M > 2, use the full batch and compute ensemble loss
    else:
        total = 0
        correct_individual = [0] * M_value
        
        for inputs, targets in tqdm(train_loader, desc=f"Training (M={M_value})"):
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Sample M inputs and targets from batch
            input_list = []
            target_list = []
            
            if batch_size >= M_value:
                indices = torch.randperm(batch_size)[:M_value]
                for i in range(M_value):
                    input_list.append(inputs[indices[i]])
                    target_list.append(targets[indices[i]])
            else:
                # If batch size is smaller than M, sample with replacement
                for _ in range(M_value):
                    idx = torch.randint(0, batch_size, (1,)).item()
                    input_list.append(inputs[idx])
                    target_list.append(targets[idx])
            
            # Expand dimensions for batch processing
            batch_inputs = torch.stack(input_list).unsqueeze(0)  # [1, M, C, H, W]
            batch_targets = torch.stack(target_list)             # [M]
            
            # Sample mixing ratios from Dirichlet distribution
            kappa_values = sample_mixing_ratios(M_value, args.alpha, device)
            
            # Extract features for each input
            features_list = []
            for i in range(M_value):
                with torch.no_grad():  # We don't need gradients for feature extraction
                    if hasattr(model, 'subnetworks'):
                        # For M>2, use the first convolutional layer of the first subnetwork
                        features_i = model.subnetworks[0].encoder(batch_inputs[:, i])
                    else:
                        # For M=2, use the encoder from the model
                        features_i = model.encoder(batch_inputs[:, i])
                features_list.append(features_i)
            
            # Apply mixing based on the approach
            use_patch_mixing = torch.rand(1).item() < mixing_prob
            
            if use_patch_mixing and args.approach in ['cut_mixmo', 'cut_mixmo_cutmix']:
                mixed_features = patch_mix_MN(features_list, kappa_values)
            else:
                mixed_features = linear_mix_MN(features_list, kappa_values)
            
            # Process the mixed features through the model 
            # and get multiple outputs (one for each subnetwork head)
            if hasattr(model, 'subnetworks'):
                # For M>2 model
                outputs = []
                for i, subnet in enumerate(model.subnetworks):
                    if i*2 + 1 < M_value:  # If we have enough outputs for this subnet
                        # Pass through the core network and get outputs from both heads
                        core_features = subnet.blockA(mixed_features)
                        core_features = subnet.blockB(core_features)
                        core_features = subnet.blockC(core_features)
                        core_features = subnet.avg_pool(core_features)
                        core_features = core_features.view(core_features.size(0), -1)
                        
                        out1 = subnet.fc1(core_features)
                        out2 = subnet.fc2(core_features)
                        
                        outputs.append(out1)
                        outputs.append(out2)
                        
                # Trim to exactly M outputs
                outputs = outputs[:M_value]
            else:
                # For M=2 model (should not reach this branch for M>2)
                outputs = []
                core_features = model.blockA(mixed_features)
                core_features = model.blockB(core_features)
                core_features = model.blockC(core_features)
                core_features = model.avg_pool(core_features)
                core_features = core_features.view(core_features.size(0), -1)
                
                outputs.append(model.fc1(core_features))
                outputs.append(model.fc2(core_features))
            
            # Calculate loss with the generalized weighting
            loss = mixmo_loss_MN(outputs, [batch_targets] * M_value, kappa_values, r=args.r_param)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate individual accuracies (for monitoring only)
            for i, output in enumerate(outputs):
                _, pred = output.max(1)
                correct_individual[i] += pred.eq(batch_targets).sum().item()
            
            total += 1  # Count batches instead of samples for simplicity
        
        # Calculate metrics
        individual_acc = [100. * c / total for c in correct_individual]
        avg_individual_acc = sum(individual_acc) / M_value
        
        return running_loss / len(train_loader), avg_individual_acc, individual_acc

def evaluate(model, test_loader, device, args, M_value, calculate_metrics=True, optimal_temp=None):
    """
    Evaluation function for all approaches.
    For M==2, it returns the ensemble accuracy and the individual accuracies;
    for M>2, it returns the ensemble accuracy and a list of individual accuracies.
    """
    model.eval()
    
    # For M=2
    if M_value == 2:
        correct_ensemble = 0
        correct1 = 0
        correct2 = 0
        total = 0
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating (M=2)"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # For inference, use the same input for both branches
                out1, out2, _, _, _ = model(inputs, inputs)
                
                # Ensemble prediction (average of both outputs)
                ensemble_output = (out1 + out2) / 2
                
                # Save logits and targets for NLL calculation
                if calculate_metrics:
                    all_logits.append(ensemble_output.detach().cpu())
                    all_targets.append(targets.detach().cpu())
                
                # Top-1 accuracy
                _, pred_ens = ensemble_output.max(1)
                correct_ensemble += pred_ens.eq(targets).sum().item()
                
                # Subnetwork accuracies
                _, pred1 = out1.max(1)
                correct1 += pred1.eq(targets).sum().item()
                
                _, pred2 = out2.max(1)
                correct2 += pred2.eq(targets).sum().item()
                
                total += targets.size(0)
        
        acc_ensemble = 100. * correct_ensemble / total
        acc1 = 100. * correct1 / total
        acc2 = 100. * correct2 / total
        individual_acc = [acc1, acc2]
        
        # Calculate NLLc if needed
        nllc = None
        if calculate_metrics and optimal_temp is not None:
            all_logits = torch.cat(all_logits)
            all_targets = torch.cat(all_targets)
            
            # Apply temperature scaling
            scaled_logits = all_logits / optimal_temp
            
            # Calculate NLLc
            nllc = calculate_nll(scaled_logits, all_targets)
        
        return acc_ensemble, individual_acc, nllc
    
    # For M > 2
    else:
        correct_ensemble = 0
        correct_individual = [0] * M_value
        total = 0
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Evaluating (M={M_value})"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # For M>2 inference with multi-network model
                if hasattr(model, 'subnetworks'):
                    outputs = []
                    for i, subnet in enumerate(model.subnetworks):
                        if i*2 + 1 < M_value:  # If we have enough outputs for this subnet
                            # Get features from the encoder
                            features = subnet.encoder(inputs)
                            
                            # Pass through the core network
                            core_features = subnet.blockA(features)
                            core_features = subnet.blockB(core_features)
                            core_features = subnet.blockC(core_features)
                            core_features = subnet.avg_pool(core_features)
                            core_features = core_features.view(core_features.size(0), -1)
                            
                            # Get outputs from both heads
                            out1 = subnet.fc1(core_features)
                            out2 = subnet.fc2(core_features)
                            
                            outputs.append(out1)
                            outputs.append(out2)
                    
                    # Trim to exactly M outputs
                    outputs = outputs[:M_value]
                else:
                    # For M=2 model (should not reach this branch for M>2)
                    out1, out2, _, _, _ = model(inputs, inputs)
                    outputs = [out1, out2]
                
                # Ensemble prediction (average of all outputs)
                ensemble_output = torch.stack(outputs).mean(dim=0)
                
                # Save logits and targets for NLL calculation
                if calculate_metrics:
                    all_logits.append(ensemble_output.detach().cpu())
                    all_targets.append(targets.detach().cpu())
                
                # Ensemble accuracy
                _, pred_ens = ensemble_output.max(1)
                correct_ensemble += pred_ens.eq(targets).sum().item()
                
                # Individual accuracies
                for i, output in enumerate(outputs):
                    _, pred = output.max(1)
                    correct_individual[i] += pred.eq(targets).sum().item()
                
                total += targets.size(0)
        
        # Calculate metrics
        acc_ensemble = 100. * correct_ensemble / total
        individual_acc = [100. * c / total for c in correct_individual]
        
        # Calculate NLLc if needed
        nllc = None
        if calculate_metrics and optimal_temp is not None:
            all_logits = torch.cat(all_logits)
            all_targets = torch.cat(all_targets)
            
            # Apply temperature scaling
            scaled_logits = all_logits / optimal_temp
            
            # Calculate NLLc
            nllc = calculate_nll(scaled_logits, all_targets)
        
        return acc_ensemble, individual_acc, nllc

def train_model_with_M_value(args, M_value, data_handler, device, num_classes):
    """Train a model with a specific M value and evaluate it"""
    print(f"\n===== Training with M = {M_value} subnetworks =====")
    
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
    model = get_model(args, num_classes, M_value).to(device)
    
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
    exp_name = f"{args.approach}_b{batch_repetitions}_wrn{args.width}_{args.dataset}_M{M_value}_run{args.run_number}"
    print(f"Starting experiment: {exp_name}")
    
    # Initialize results storage
    results = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_acc_subnet': []  # For storing individual subnet accuracies
    }
    
    best_acc = 0.0
    best_epoch = 0
    final_metrics = {}
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train the model with the specific M value
        train_loss, train_acc, train_acc_individual = train_epoch(
            model, train_loader, optimizer, device, args, 
            M_value=M_value, epoch=epoch, total_epochs=args.epochs
        )
        scheduler.step()
        
        # Regular evaluation for training monitoring (without calculating NLLc)
        test_acc, test_acc_individual, _ = evaluate(
            model, test_loader, device, args,
            M_value=M_value,
            calculate_metrics=False
        )
        
        # Log regular results
        results['epoch'].append(epoch)
        results['train_loss'].append(float(train_loss))
        results['train_acc'].append(float(train_acc))
        results['test_acc'].append(float(test_acc))
        results['test_acc_subnet'].append([float(acc) for acc in test_acc_individual])
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Ensemble Acc: {test_acc:.2f}%")
        print(f"Individual Accuracies: {[f'{acc:.2f}%' for acc in test_acc_individual]}")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            
            # Save model
            model_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
            
        # For Figure 9, we want to collect metrics for each epoch
        if 'final_metrics' not in final_metrics:
            final_metrics['final_metrics'] = []
        
        final_metrics['final_metrics'].append({
            'ensemble_acc': float(test_acc),
            'individual_acc': [float(acc) for acc in test_acc_individual]
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
    optimal_temp = optimize_temperature(model, calib_loader, device, args, M_value)
    print(f"Optimal temperature: {optimal_temp:.4f}")
    
    # Final evaluation with temperature scaling on evaluation set
    test_acc, test_acc_individual, nllc = evaluate(
        model, eval_loader, device, args,
        M_value=M_value,
        optimal_temp=optimal_temp
    )
    
    # Update results with final values
    results['final_acc_ensemble'] = float(test_acc)
    results['final_acc_subnet'] = [float(acc) for acc in test_acc_individual]
    results['final_nllc'] = float(nllc) if nllc is not None else None
    results['best_acc'] = float(best_acc)
    results['best_epoch'] = best_epoch
    results['optimal_temperature'] = float(optimal_temp)
    
    # Save detailed results for this M value
    results_path = os.path.join(args.save_dir, f"{exp_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Training completed. Best test accuracy: {best_acc:.2f}% at epoch {best_epoch+1}")
    print(f"Final metrics - Ensemble acc: {test_acc:.2f}%, Subnet acc: {[f'{acc:.2f}%' for acc in test_acc_individual]}")
    
    # Calculate average metrics from the last 10 epochs for Figure 9
    last_10_epochs = final_metrics['final_metrics'][-10:]
    avg_ensemble_acc = np.mean([m['ensemble_acc'] for m in last_10_epochs])
    
    # Average individual accuracies across all subnets and last 10 epochs
    avg_individual_acc = np.mean([np.mean(m['individual_acc']) for m in last_10_epochs])
    
    avg_metrics = {
        'M': M_value,
        'ensemble_acc': float(avg_ensemble_acc),
        'individual_acc': float(avg_individual_acc)
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
    
    # Figure 9 experiment: vary M value and measure performance
    # Parse M values from command line argument
    M_values = [int(M) for M in args.M_values.split(',')]
    figure9_results = []
    
    for M_value in M_values:
        # Set seed based on run number and reset for each M value
        seed = args.seed + args.run_number - 1
        set_seed(seed)
        
        # Train and evaluate model with this M value
        avg_metrics = train_model_with_M_value(
            args, M_value, data_handler, device, num_classes
        )
        
        figure9_results.append(avg_metrics)
    
    # Save the combined results for Figure 9
    figure9_path = os.path.join(args.save_dir, f"figure9_results_{args.approach}_run{args.run_number}.json")
    with open(figure9_path, 'w') as f:
        json.dump(figure9_results, f, indent=4)
    
    print(f"Figure 9 experiment completed. Results saved to {figure9_path}")
    
    # Print summary of results
    print("\nSummary of results for different M values:")
    print("M\tEnsemble Acc\tAvg Individual Acc")
    for result in figure9_results:
        print(f"{result['M']}\t{result['ensemble_acc']:.2f}%\t{result['individual_acc']:.2f}%")

if __name__ == "__main__":
    main()
