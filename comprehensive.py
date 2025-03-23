import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable


#####################################
# MixMo Trainer Class
#####################################

class MixMoTrainer:
    """
    Trainer class for MixMo models that handles training and evaluation procedures.
    This class also integrates various metrics for accuracy, calibration, and diversity.
    """
    
    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        checkpoint_dir: str = 'checkpoints',
    ):
        """
        Initialize the trainer.
        
        Args:
            model: MixMo model to train (e.g., LinearMixMo, CutMixMo, etc.)
            optimizer: Optimizer for training.
            scheduler: Optional learning rate scheduler.
            device: Device to train on.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist.
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Move model to device.
        self.model.to(self.device)
    
    def train_step(
        self, 
        x_batch: torch.Tensor, 
        y_batch: torch.Tensor,
        batch_repetition: int = 1
    ) -> Dict[str, float]:
        """
        Perform a single training step with batch repetition as described in the MixMo paper.
        
        Args:
            x_batch: Batch of input images.
            y_batch: Batch of labels.
            batch_repetition: Number of times each sample appears (b in the paper).
                With b=1, each sample appears once paired with one random sample.
                With b>1, each sample appears b times paired with different random samples.
            
        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        self.optimizer.zero_grad()
    
        batch_size = x_batch.shape[0]
        total_loss = 0.0
        total_acc0 = 0.0
        total_acc1 = 0.0
    
        # For each repetition, create a new permutation and process the batch.
        for b_idx in range(batch_repetition):
            # Create a random permutation for this repetition.
            perm = torch.randperm(batch_size, device=self.device)
        
            # Create pairs: original samples with permuted samples.
            x0, y0 = x_batch, y_batch            # Original samples.
            x1, y1 = x_batch[perm], y_batch[perm]  # Permuted samples.
        
            # Create input by concatenating each sample with its permuted pair.
            x_concat = torch.cat([x0, x1], dim=1)
        
            # Forward pass.
            pred0, pred1, lam = self.model(x_concat, {"mode": "train"})
        
            # Compute loss for this batch permutation.
            batch_loss = self.model.compute_loss(pred0, pred1, y0, y1, lam)
        
            # Backward pass.
            batch_loss.backward()
            total_loss += batch_loss.item()
        
            # Calculate accuracy for both subnetworks.
            acc0 = (pred0.argmax(dim=1) == y0).float().mean().item()
            acc1 = (pred1.argmax(dim=1) == y1).float().mean().item()
            total_acc0 += acc0 / batch_repetition
            total_acc1 += acc1 / batch_repetition
        
        # Adjust learning rate for batch repetition (temporary scaling).
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= batch_repetition 
    
        # Optimizer step.
        self.optimizer.step()
            
        # Restore the original learning rate.
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= batch_repetition
        
        # Update learning rate scheduler if provided.
        if self.scheduler is not None:
            self.scheduler.step()
    
        metrics = {
            "loss": total_loss / batch_repetition,
            "acc0": total_acc0,
            "acc1": total_acc1,
            "acc_avg": (total_acc0 + total_acc1) / 2
        } 
    
        return metrics
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        batch_repetition: int = 1,
        verbose: bool = True,
        log_interval: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data.
            batch_repetition: Number of times to repeat samples in a batch.
            verbose: Whether to print progress.
            log_interval: How often to log progress.
            
        Returns:
            Dictionary with epoch metrics.
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_start_time = time.time()
        steps = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Perform training step.
            metrics = self.train_step(inputs, targets, batch_repetition)
            
            # Update epoch metrics.
            epoch_loss += metrics["loss"]
            epoch_acc += metrics["acc_avg"]
            steps += 1
            
            # Print progress.
            if verbose and batch_idx % log_interval == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {metrics['loss']:.4f}, Acc: {metrics['acc_avg']:.4f}")
        
        # Calculate average metrics for the epoch.
        epoch_loss /= steps
        epoch_acc /= steps
        epoch_time = time.time() - epoch_start_time
        
        epoch_metrics = {
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "epoch_time": epoch_time
        }
        
        return epoch_metrics
    
    def evaluate(
        self, 
        data_loader: DataLoader,
        temperature_scaling: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation.
            temperature_scaling: Whether to apply temperature scaling for calibration.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        
        correct = 0
        total = 0
        loss_sum = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Duplicate input for inference.
                x_double = torch.cat([inputs, inputs], dim=1)
                
                # Forward pass.
                outputs = self.model(x_double, {"mode": "inference"})
                
                # Calculate loss.
                loss = nn.CrossEntropyLoss()(outputs, targets)
                predicted = outputs.argmax(dim=1)
                
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                loss_sum += loss.item() * targets.size(0)
                
                # Store outputs and targets for additional metrics.
                all_outputs.append(outputs)
                all_targets.append(targets)
        
        # Concatenate outputs and targets.
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        
        # Calculate calibrated NLL if requested.
        if temperature_scaling:
            temperature = self._find_optimal_temperature(all_outputs, all_targets)
            calibrated_nll, _ = self._calculate_calibrated_nll(all_outputs, all_targets, temperature)
        else:
            calibrated_nll = nn.CrossEntropyLoss()(all_outputs, all_targets).item()
        
        accuracy = correct / total
        avg_loss = loss_sum / total
        
        # Calculate Top-5 accuracy if applicable.
        if self.model.num_classes >= 5:
            top5_correct = 0
            _, top5_pred = all_outputs.topk(5, 1, True, True)
            for i in range(all_targets.size(0)):
                if all_targets[i] in top5_pred[i]:
                    top5_correct += 1
            top5_accuracy = top5_correct / total
        else:
            top5_accuracy = accuracy
        
        # Optionally, compute diversity metrics as well.
        diversity_metrics = calculate_diversity_metrics(self.model, data_loader, self.device)
        
        eval_metrics = {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy,
            "eval_top5_accuracy": top5_accuracy,
            "eval_nllc": calibrated_nll,
            # Merge in diversity metrics.
            **diversity_metrics
        }
        
        return eval_metrics
    
    def _find_optimal_temperature(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Find optimal temperature for temperature scaling.
        
        Args:
            logits: Model outputs.
            targets: Ground truth labels.
            
        Returns:
            Optimal temperature as a tensor.
        """
        temperatures = torch.linspace(0.5, 3.0, 50).to(self.device)
        best_nll = float('inf')
        best_temp = 1.0
        
        with torch.no_grad():
            for temp in temperatures:
                scaled_logits = torch.log_softmax(logits / temp, dim=1)
                nll = nn.CrossEntropyLoss()(scaled_logits, targets).item()
                if nll < best_nll:
                    best_nll = nll
                    best_temp = temp.item()
        
        return torch.tensor(best_temp, device=self.device)
    
    def _calculate_calibrated_nll(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        temperature: float
    ) -> Tuple[float, float]:
        """
        Calculate NLL with temperature scaling.
        
        Args:
            logits: Model outputs.
            targets: Ground truth labels.
            temperature: Temperature for scaling.
            
        Returns:
            Tuple of (calibrated NLL, best temperature).
        """
        with torch.no_grad():
            scaled_logits = torch.log_softmax(logits / temperature, dim=1)
            nll = nn.CrossEntropyLoss()(scaled_logits, targets).item()
        return nll, temperature
    
    def calculate_ece(
        self, 
        data_loader: DataLoader, 
        num_bins: int = 15
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            data_loader: DataLoader for evaluation.
            num_bins: Number of bins for the histogram.
            
        Returns:
            ECE value.
        """
        self.model.eval()
        confidences = []
        predictions = []
        targets = []
        
        with torch.no_grad():
            for inputs, batch_targets in data_loader:
                inputs, batch_targets = inputs.to(self.device), batch_targets.to(self.device)
                x_double = torch.cat([inputs, inputs], dim=1)
                outputs = self.model(x_double, {"mode": "inference"})
                softmax_outputs = torch.softmax(outputs, dim=1)
                confidence, prediction = torch.max(softmax_outputs, dim=1)
                confidences.append(confidence)
                predictions.append(prediction)
                targets.append(batch_targets)
        
        confidences = torch.cat(confidences).cpu().numpy()
        predictions = torch.cat(predictions).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                bin_acc = np.mean(predictions[in_bin] == targets[in_bin])
                bin_conf = np.mean(confidences[in_bin])
                ece += np.abs(bin_acc - bin_conf) * prop_in_bin
        return ece
    
    def train(
        self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        batch_repetition: int = 1,
        save_best: bool = True,
        save_interval: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            num_epochs: Number of epochs to train.
            batch_repetition: Number of times to repeat samples in a batch.
            save_best: Whether to save the best model.
            save_interval: How often to save checkpoints.
            verbose: Whether to print progress.
            
        Returns:
            Dictionary with training history.
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_nllc": []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            if verbose:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch.
            train_metrics = self.train_epoch(
                train_loader=train_loader,
                batch_repetition=batch_repetition,
                verbose=verbose
            )
            
            # Evaluate on validation set.
            val_metrics = self.evaluate(val_loader, temperature_scaling=True)
            
            history["train_loss"].append(train_metrics["train_loss"])
            history["train_acc"].append(train_metrics["train_acc"])
            history["val_loss"].append(val_metrics["eval_loss"])
            history["val_acc"].append(val_metrics["eval_accuracy"])
            history["val_nllc"].append(val_metrics["eval_nllc"])
            
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Train Acc: {train_metrics['train_acc']:.4f}, Val Loss: {val_metrics['eval_loss']:.4f}, "
                      f"Val Acc: {val_metrics['eval_accuracy']:.4f}, Val NLLc: {val_metrics['eval_nllc']:.4f}")
            
            if save_best and val_metrics["eval_accuracy"] > best_val_acc:
                best_val_acc = val_metrics["eval_accuracy"]
                self.save_checkpoint(f"{self.checkpoint_dir}/best_model.pth")
                if verbose:
                    print(f"New best model saved with accuracy: {best_val_acc:.4f}")
            
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"{self.checkpoint_dir}/model_epoch_{epoch+1}.pth")
        
        self.save_checkpoint(f"{self.checkpoint_dir}/final_model.pth")
        return history
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_classes": self.model.num_classes,
            "alpha": self.model.alpha,
            "r": self.model.r
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

#####################################
# 5. Metrics and Utilities
#####################################

class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_diversity_metrics(
    model,
    dataloader,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Dict[str, float]:
    """
    Calculate diversity metrics between the two subnetworks in a MixMo model.
    
    Returns a dictionary containing:
        - accuracy_subnet0: Accuracy of first subnetwork.
        - accuracy_subnet1: Accuracy of second subnetwork.
        - both_correct_rate: Proportion of samples correctly predicted by both subnetworks.
        - both_wrong_rate: Proportion of samples misclassified by both.
        - different_errors: Proportion of samples where one subnetwork is correct and the other is wrong.
        - disagreement_rate: Percentage of samples with different predictions.
        - ratio_error: Ratio between different errors and simultaneous errors.
        - average_ensemble_acc: Average accuracy of both subnetworks.
        - true_ensemble_acc: Accuracy of the ensemble prediction.
    """
    model.eval()
    correct0 = 0
    correct1 = 0
    both_correct = 0
    both_wrong = 0
    different_errors = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get features from both encoders.
            l0 = model.encoder0(inputs)
            l1 = model.encoder1(inputs)
            
            # Pass through backbone.
            features0 = model.backbone(l0)
            features1 = model.backbone(l1)
            
            # Get predictions from both heads.
            pred0 = model.head0(features0)
            pred1 = model.head1(features1)
            
            predicted0 = pred0.argmax(dim=1)
            predicted1 = pred1.argmax(dim=1)
            
            is_correct0 = (predicted0 == targets)
            is_correct1 = (predicted1 == targets)
            
            correct0 += is_correct0.sum().item()
            correct1 += is_correct1.sum().item()
            both_correct += (is_correct0 & is_correct1).sum().item()
            both_wrong += ((~is_correct0) & (~is_correct1)).sum().item()
            different_errors += (((~is_correct0) & is_correct1) | (is_correct0 & (~is_correct1))).sum().item()
            total += targets.size(0)
    
    acc0 = correct0 / total
    acc1 = correct1 / total
    disagreement = (predicted0 != predicted1).sum().item() / total
    ratio_error = different_errors / (both_wrong + 1e-10)
    ensemble_acc = (correct0 + correct1) / (2 * total)
    
    # Ensemble prediction from duplicated input.
    inputs_doubled = torch.cat([inputs, inputs], dim=1)
    ensemble_pred = model(inputs_doubled, {"mode": "inference"})
    ensemble_pred_class = ensemble_pred.argmax(dim=1)
    ensemble_correct = (ensemble_pred_class == targets).sum().item()
    true_ensemble_acc = ensemble_correct / total
    
    return {
        "accuracy_subnet0": acc0,
        "accuracy_subnet1": acc1,
        "both_correct_rate": both_correct / total,
        "both_wrong_rate": both_wrong / total,
        "different_errors": different_errors / total,
        "disagreement_rate": disagreement,
        "ratio_error": ratio_error,
        "average_ensemble_acc": ensemble_acc,
        "true_ensemble_acc": true_ensemble_acc
    }


def compute_reweighting_function(lam_values: np.ndarray, r: int) -> np.ndarray:
    """
    Compute the reweighting function wr for visualization.
    
    Formula: wr(λ) = 2 * λ^(1/r) / (λ^(1/r) + (1-λ)^(1/r))
    
    Args:
        lam_values: Array of λ values.
        r: Reweighting parameter.
        
    Returns:
        Array of weight values.
    """
    eps = 1e-7
    lam = np.clip(lam_values, eps, 1 - eps)
    lam_r = np.power(lam, 1.0 / r)
    inv_lam_r = np.power(1.0 - lam, 1.0 / r)
    weights = 2 * lam_r / (lam_r + inv_lam_r)
    return weights


def plot_reweighting_functions(r_values: List[int], num_points: int = 100) -> None:
    """
    Plot the reweighting functions for different r values.
    """
    lam = np.linspace(0.01, 0.99, num_points)
    plt.figure(figsize=(10, 6))
    for r in r_values:
        weights = compute_reweighting_function(lam, r)
        plt.plot(lam, weights, label=f'r = {r}')
    plt.plot(lam, lam, '--', label='Identity (λ→λ)')
    plt.plot(lam, np.ones_like(lam), ':', label='Constant (λ→1)')
    plt.xlabel('λ')
    plt.ylabel('wr(λ)')
    plt.title('Reweighting Functions for Different r Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reweighting_functions.png')
    plt.close()


def calculate_calibrated_nll(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """
    Calculate calibrated NLL using temperature scaling.
    
    Returns a tuple of (calibrated NLL, best temperature).
    """
    temperatures = torch.linspace(0.1, 5.0, 50)
    best_nll = float('inf')
    best_temp = 1.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for temp in temperatures:
            scaled_logits = logits / temp
            nll = criterion(scaled_logits, targets).item()
            if nll < best_nll:
                best_nll = nll
                best_temp = temp.item()
    
    scaled_logits = logits / best_temp
    calibrated_nll = criterion(scaled_logits, targets).item()
    return calibrated_nll, best_temp


def calculate_ece(confidences: np.ndarray, predictions: np.ndarray, 
                  targets: np.ndarray, num_bins: int = 15) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            bin_acc = np.mean(predictions[in_bin] == targets[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += np.abs(bin_acc - bin_conf) * prop_in_bin
    return ece


def plot_training_history(history: Dict[str, List[float]], output_path: str = 'training_history.png') -> None:
    """
    Plot training history: loss and accuracy curves.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_diversity_metrics(subnets_accuracy: List[float], ensemble_accuracy: List[float], 
                           ratio_error: List[float], output_path: str = 'diversity_metrics.png') -> None:
    """
    Plot diversity metrics (subnet accuracies, ensemble accuracy, and ratio-error) against a width parameter.
    """
    width_values = range(2, 2 + len(subnets_accuracy))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(width_values, subnets_accuracy, 'b-', label='Subnet Accuracy')
    ax1.plot(width_values, ensemble_accuracy, 'r-', label='Ensemble Accuracy')
    ax1.set_title('Accuracy vs. Width')
    ax1.set_xlabel('Width Parameter')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(width_values, ratio_error, 'g-', label='Ratio-Error (dre)')
    ax2.set_title('Diversity (Ratio-Error) vs. Width')
    ax2.set_xlabel('Width Parameter')
    ax2.set_ylabel('Ratio-Error')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
