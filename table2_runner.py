import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from tqdm import tqdm
import json
from data_handler import DataHandler
from models import Wide_ResNet28
from MixMo import cut_mixmo
# Add this at the beginning of your main() function
def debug_cut_mixmo():
    # Create some test tensors
    features1 = torch.randn(4, 16, 8, 8)  # Example size
    features2 = torch.randn(4, 16, 8, 8)

    # Call the function and check what it returns
    result = cut_mixmo(features1, features2, alpha=2.0)
    print(f"cut_mixmo returns {len(result)} values")
    print(f"Type of result: {type(result)}")

    if isinstance(result, tuple):
        for i, val in enumerate(result):
            print(f"  Result[{i}] type: {type(val)}")

    return result

# Call the debug function
debug_result = debug_cut_mixmo()
print("Debug complete")
def main():
  # Add at the beginning of your main function
    torch.cuda.empty_cache()  # Clear any cached memory
    # Configuration for CutMix approach
    #batch_size = 128
    batch_size=64
    batch_repetitions = 2
    width = 10
    epochs = 300
    alpha = 2.0
    r = 3  # Reweighting parameter
    data_root = './data'
    save_dir = './results'

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get CIFAR-100 dataset
    data_handler = DataHandler(data_root=data_root)
    train_loader, test_loader = data_handler.get_cifar100(
        batch_size=batch_size, batch_repetitions=batch_repetitions)

    # Create the model with CutMix approach
    model = Wide_ResNet28(
        widen_factor=width,
        dropout_rate=0.3,
        num_classes=100,
        augmentation_type='cutmixmo'
    )
    model.to(device)

    # Create optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=3e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 200, 225], gamma=0.1)

    # Training loop
    print("Starting training with CutMix approach")
    best_acc = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Calculate the current mixing probability (linear decay in last 1/12 of training)
        mixing_prob = 0.5
        if epoch > epochs - (epochs // 12):
            mixing_prob = 0.5 * (1 - (epoch - (epochs - (epochs // 12))) / (epochs // 12))

        # Training
        model.train()
        train_loss = 0
        correct1 = 0
        correct2 = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc="Training"):
            # Split batch for repetition
            batch_size = inputs.size(0) // batch_repetitions
            x_chunks = torch.chunk(inputs, batch_repetitions)
            y_chunks = torch.chunk(targets, batch_repetitions)

            # Take first two chunks
            x1, x2 = x_chunks[0].to(device), x_chunks[1].to(device)
            y1, y2 = y_chunks[0].to(device), y_chunks[1].to(device)

            # Set mixing mode (patch or linear)
            use_patch_mixing = torch.rand(1).item() < mixing_prob
            if hasattr(model, 'set_mixing_mode'):
                model.set_mixing_mode('patch' if use_patch_mixing else 'linear')

            # Forward pass
            out1, out2, out_mix1, out_mix2, kappa = model(x1, x2)

            # Compute weights for loss
            kappa_pow = kappa ** (1/r)
            inv_kappa_pow = (1 - kappa) ** (1/r)
            weights = 2 * kappa_pow / (kappa_pow + inv_kappa_pow)

            # Calculate loss
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss1 = criterion(out_mix1, y1)
            loss2 = criterion(out_mix2, y2)

            loss = (weights * loss1).mean() + ((2 - weights) * loss2).mean()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            _, pred1 = out1.max(1)
            _, pred2 = out2.max(1)
            correct1 += pred1.eq(y1).sum().item()
            correct2 += pred2.eq(y2).sum().item()
            total += y1.size(0)

        scheduler.step()

        # Evaluation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating"):
                inputs, targets = inputs.to(device), targets.to(device)

                # Duplicate input for inference
                out1, out2, _, _, _ = model(inputs, inputs)

                # Average predictions
                outputs = (out1 + out2) / 2

                # Calculate accuracy
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        # Calculate metrics
        train_acc = 100. * (correct1 + correct2) / (2 * total)
        test_acc = 100. * correct / total

        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"{save_dir}/cutmix_best.pth")
            print(f"New best model saved with accuracy: {best_acc:.2f}%")

    # Calculate NLLc using temperature scaling
    print("\nCalculating NLLc with temperature scaling...")

    # Split test set for calibration and evaluation
    test_indices = list(range(len(test_loader.dataset)))
    np.random.shuffle(test_indices)
    split = len(test_indices) // 2

    from torch.utils.data import Subset, DataLoader
    calib_dataset = Subset(test_loader.dataset, test_indices[:split])
    eval_dataset = Subset(test_loader.dataset, test_indices[split:])

    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Find optimal temperature
    model.load_state_dict(torch.load(f"{save_dir}/cutmix_best.pth"))
    model.eval()

    # Collect logits and labels from calibration set
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(calib_loader, desc="Collecting calibration data"):
            inputs, targets = inputs.to(device), targets.to(device)
            out1, out2, _, _, _ = model(inputs, inputs)
            outputs = (out1 + out2) / 2
            all_logits.append(outputs)
            all_labels.append(targets)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Find optimal temperature
    temperatures = torch.linspace(0.5, 3.0, 50).to(device)
    best_nll = float('inf')
    best_temp = 1.0

    criterion = nn.CrossEntropyLoss()

    for temp in temperatures:
        scaled_logits = all_logits / temp
        nll = criterion(scaled_logits, all_labels).item()
        if nll < best_nll:
            best_nll = nll
            best_temp = temp.item()

    print(f"Optimal temperature: {best_temp}")

    # Calculate NLLc on evaluation set
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(eval_loader, desc="Evaluating with calibration"):
            inputs, targets = inputs.to(device), targets.to(device)
            out1, out2, _, _, _ = model(inputs, inputs)
            outputs = (out1 + out2) / 2
            all_logits.append(outputs)
            all_labels.append(targets)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Apply temperature scaling
    scaled_logits = all_logits / best_temp
    nllc = criterion(scaled_logits, all_labels).item()

    # Final results
    print("\nFinal CutMix results:")
    print(f"Top1 (↑): {best_acc:.2f}%")
    print(f"NLLc (↓): {nllc:.3f}")

    # Save results
    results = {
        "approach": "CutMix",
        "top1": float(best_acc),
        "nllc": float(nllc)
    }

    with open(f"{save_dir}/cutmix_table2_results.json", 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {save_dir}/cutmix_table2_results.json")

if __name__ == "__main__":
    main()
