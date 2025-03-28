import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import Subset, DataLoader
from data_handler import DataHandler
from models import Wide_ResNet28

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def get_model(num_classes):
    """Create standard Wide_ResNet model without MixMo augmentation"""
    return Wide_ResNet28(
        widen_factor=10,  # WRN-28-10 architecture
        dropout_rate=0.3,
        num_classes=num_classes,
        augmentation_type='none',  # Standard model, not MixMo
    )

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.temperature

def optimize_temperature(model, val_loader, device):
    temperature_model = TemperatureScaling().to(device)
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([temperature_model.temperature], lr=0.01, max_iter=50)

    logits_list = []
    labels_list = []

    with torch.no_grad():
        model.eval()
        for inputs, targets in tqdm(val_loader, desc="Collecting logits for calibration"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            logits_list.append(outputs.detach().cpu())
            labels_list.append(targets.detach().cpu())

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    def eval():
        optimizer.zero_grad()
        scaled_logits = temperature_model(logits)
        loss = nll_criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    return temperature_model.temperature.item()

def calculate_nll(logits, targets):
    probs = torch.nn.functional.softmax(logits, dim=1)
    correct_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()
    nll = -torch.log(correct_probs)
    return nll.mean().item()

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Handle the different output formats
        # Check if outputs is a tuple (which might be happening with your model)
        if isinstance(outputs, tuple):
            # Use the first item of the tuple as the main output
            outputs = outputs[0]

        # Calculate loss - handle one-hot encoded targets from CutMix
        if targets.dim() > 1 and targets.size(1) > 1:
            # CutMix returns soft targets
            loss = torch.sum(-targets * torch.log_softmax(outputs, dim=1), dim=1).mean()
            # Get the original class labels for accuracy calculation
            true_targets = targets.argmax(dim=1)
        else:
            # Standard targets
            loss = criterion(outputs, targets)
            true_targets = targets

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct += predicted.eq(true_targets).sum().item()
        total += targets.size(0)

    acc = 100. * correct / total

    return running_loss / len(train_loader), acc

def evaluate(model, test_loader, device, calculate_metrics=True, optimal_temp=None):
    model.eval()

    correct = 0
    total = 0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Standard forward pass
            outputs = model(inputs)

            # Handle the different output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Save logits and targets for NLL calculation
            if calculate_metrics:
                all_logits.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())

            # Handle targets that might be one-hot encoded
            if targets.dim() > 1 and targets.size(1) > 1:
                true_targets = targets.argmax(dim=1)
            else:
                true_targets = targets

            # Top-1 accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(true_targets).sum().item()
            total += targets.size(0)

    acc = 100. * correct / total

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

    return acc, nllc

def main():
    # Hyperparameters as specified for Table 2
    batch_size = 64
    batch_repetitions = 2  # b=2 as specified
    effective_batch_size = batch_size * batch_repetitions
    dataset = 'cifar100'
    num_classes = 100
    epochs = 300
    alpha = 1.0  # Default alpha for CutMix

    # Set seed for reproducibility
    set_seed(42)

    # Create save directory
    save_dir = './WRN_CIFAR_Results'
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data handler
    data_handler = DataHandler(data_root='./data')

    # Get CIFAR-100 dataset and loaders
    # First get the standard dataset without batch repetition
    train_loader, test_loader = data_handler.get_cifar100(
        batch_size=batch_size,
        batch_repetitions=1  # Important: first get standard dataset with no repetition
    )

    # Store the original dataset
    train_dataset = train_loader.dataset

    # Now create a CutMix loader with the correct dataset and effective batch size
    train_loader = data_handler.get_cutmix_loader(
        dataset=train_dataset,
        batch_size=effective_batch_size,
        alpha=alpha,
        num_classes=num_classes,
        batch_repetitions=batch_repetitions  # Apply batch repetition in CutMix loader
    )

    # Create and configure model - WRN-28-10
    model = get_model(num_classes).to(device)

    # Learning rate adjusted for batch repetition
    initial_lr = 0.1 * (batch_size / 128)
    optimizer = optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=0.9,
        weight_decay=3e-4
    )

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 225], gamma=0.1)

    # Generate experiment name
    exp_name = f"cutmix_wrn28-10_cifar100_b{batch_repetitions}"
    print(f"Starting experiment: {exp_name}")
    print(f"Epochs: {epochs}, Effective batch size: {effective_batch_size}")

    # Initialize results storage
    results = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_nllc': []
    }

    best_acc = 0.0
    best_epoch = 0

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Train the model
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device
        )
        scheduler.step()

        # Regular evaluation without NLLc for training monitoring
        test_acc, _ = evaluate(
            model, test_loader, device,
            calculate_metrics=False
        )

        # Log regular results
        results['epoch'].append(epoch)
        results['train_loss'].append(float(train_loss))
        results['train_acc'].append(float(train_acc))
        results['test_acc'].append(float(test_acc))
        results['test_nllc'].append(None)  # For now

        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

            # Save model
            model_path = os.path.join(save_dir, f"{exp_name}_best.pth")
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
        batch_size=batch_size,
        shuffle=False
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Load the best model
    model_path = os.path.join(save_dir, f"{exp_name}_best.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Optimize temperature on calibration set
    print("Optimizing temperature for calibration...")
    optimal_temp = optimize_temperature(model, calib_loader, device)
    print(f"Optimal temperature: {optimal_temp:.4f}")

    # Final evaluation with temperature scaling on evaluation set
    test_acc, nllc = evaluate(
       model, eval_loader, device,
       optimal_temp=optimal_temp
    )

    # Print final results for Table 2
    print("\n--- FINAL RESULTS FOR TABLE 2 ---")
    print(f"MODEL: CutMix (WRN-28-10)")
    print(f"TOP1 ACCURACY: {test_acc:.2f}%")
    print(f"NLLc: {nllc:.3f}")

    # Update results with final NLLc value
    if nllc is not None:
       results['final_nllc'] = float(nllc)
       print(f"Final NLLc: {nllc:.4f}")

    # Save final results
    results['best_acc'] = float(best_acc)
    results['best_epoch'] = best_epoch
    results['optimal_temperature'] = float(optimal_temp)
    results['final_test_acc'] = float(test_acc)

    results_path = os.path.join(save_dir, f"{exp_name}_results.json")
    with open(results_path, 'w') as f:
       json.dump(results, f, indent=4)

    print(f"Training completed. Best test accuracy: {best_acc:.2f}% at epoch {best_epoch+1}")
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
   main()
