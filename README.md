# Advanced Data Augmentation for Deep Learning

This repository provides implementations for advanced data augmentation techniques for deep learning models, including CutMix and MixMo.

## Installation

```bash
pip install torch torchvision
pip install datasets  # for TinyImageNet
```

## DataHandler Usage

The `DataHandler` class provides easy access to datasets with various augmentation techniques.

### Basic Usage

```python
from data_handler import DataHandler

# Initialize DataHandler
data_handler = DataHandler(data_root='./data')

# Load CIFAR-10 with standard augmentations
train_loader, test_loader = data_handler.get_cifar10(batch_size=128)

# Load CIFAR-100 with standard augmentations
train_loader, test_loader = data_handler.get_cifar100(batch_size=128)

# Load TinyImageNet with standard augmentations
train_loader, val_loader = data_handler.get_tiny_imagenet(batch_size=128)

#Load CIFAR-100-C for robustness test
available_corruptions = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform',
    'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur',
    'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate',
    'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur'
]

corruption_name = "gaussian_noise"
cifar100c_loader = data_handler.get_cifar100c(
    corruption_type=corruption_name,
    batch_size=128,
    severity=3  # Default is already 3, but specified for clarity
)
```

### Using CutMix

CutMix augmentation is implemented using the official `torchvision.transforms.v2.CutMix` transform:

```python
# Get a dataset first
data_handler = DataHandler()
train_dataset, _ = data_handler.get_cifar10()

# Create a CutMix DataLoader
cutmix_loader = data_handler.get_cutmix_loader(
    dataset=train_dataset,
    batch_size=128,
    alpha=1.0,     # Parameter for Beta distribution
    num_classes=10 # Number of classes in the dataset
)

# Training with CutMix
for images, labels in cutmix_loader:
    # labels are now one-hot encoded: [batch_size, num_classes]
    outputs = model(images)
    loss = criterion(outputs, labels)
    # ... rest of training loop
```

### Using AugMix

AugMix augmentation is implemented using the official `torchvision.transforms.AugMix` transform:

```python
# Get a dataset first
data_handler = DataHandler()
train_dataset, _ = data_handler.get_cifar10()

# Create an AugMix DataLoader
augmix_loader = data_handler.get_augmix_loader(
    dataset=train_dataset,
    batch_size=128,
    severity=3,        # Severity of base augmentations
    mixture_width=3,   # Number of augmentation chains
    chain_depth=-1,    # Random depth between 1-3
    alpha=1.0          # Hyperparameter for probability distributions
)

# Training with AugMix
for images, labels in augmix_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)
    # ... rest of training loop
```

### Batch Repetition (for MixMo)

For MixMo experiments, you can use batch repetition parameter to generate batches where each sample appears multiple times:

```python
# Create DataLoader with batch repetition (b=2)
train_loader, _ = data_handler.get_cifar10(batch_size=128, batch_repetitions=2)

# This creates batches where each unique sample appears twice
# For batch_size=128 and batch_repetitions=2, you'll get 64 unique samples per batch,
# each repeated 2 times consecutively
```

## Implementation Details

### CutMix vs CutMixMo

#### CutMix
CutMix combines two images by replacing a random patch in one image with a patch from another image:

```python
from torchvision.transforms import v2
import torch

# Initialize CutMix transform
cutmix = v2.CutMix(num_classes=10, alpha=1.0)

# Apply CutMix to a batch
images, labels = next(iter(train_loader))
mixed_images, mixed_labels = cutmix(images, labels)

# mixed_labels are now one-hot encoded with soft labels
```

#### CutMixMo

CutMixMo extends the CutMix concept to feature embeddings in a network. It is typically used with multi-branch networks where the outputs of parallel branches are mixed:

```python
import torch
import math

def cut_mixmo(l0, l1, alpha=2.0):
    """
    Implementation of Cut-MixMo augmentation from the paper:
    "MixMo: Mixing Multiple Inputs for Multiple Outputs via Deep Subnetworks"
    
    This implements equation (2) from the paper:
    M_Cut-MixMo(l0, l1) = 2[1_M ⊙ l0 + (1 - 1_M) ⊙ l1]
    
    Where 1_M is a binary mask with area ratio κ ~ Beta(α, α).
    
    Args:
        l0: First feature embedding tensor [batch_size, channels, height, width]
        l1: Second feature embedding tensor [batch_size, channels, height, width]
        alpha: Parameter for Beta distribution (default 2.0)
        
    Returns:
        Mixed feature embeddings and the mixing ratios κ used
    """
    # Get tensor dimensions
    batch_size, channels, height, width = l0.shape
    device = l0.device
    
    # Sample mixing ratio kappa from Beta(α, α) - The area to be mixed
    kappa = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(device)
    
    # Create binary masks for each example in the batch
    masks = []
    for i in range(batch_size):
        # Calculate patch area based on kappa - if it was 20%, based on h and w calculate shape
        patch_area = kappa[i].item() * height * width
        
        # Calculate patch dimensions (approximately square) - to simplify 
        patch_height = int(math.sqrt(patch_area))
        patch_width = int(patch_area / patch_height)
        
        # Ensure patch dimensions are valid - so we don't go outside shape
        patch_height = max(1, min(patch_height, height))
        patch_width = max(1, min(patch_width, width))
        
        # Sample random location for the patch
        top = torch.randint(0, height - patch_height + 1, (1,)).item()
        left = torch.randint(0, width - patch_width + 1, (1,)).item()
        
        # Create binary mask (initialized with zeros)
        mask = torch.zeros(height, width, device=device)
        
        # Set the patch area to 1
        mask[top:top+patch_height, left:left+patch_width] = 1
        
        # Randomly decide whether to use the mask or its complement
        if torch.rand(1).item() > 0.5:
            mask = 1 - mask
            
        masks.append(mask)
    
    # Stack masks and add channel dimension
    binary_mask = torch.stack(masks)[:, None, :, :]
    
    # Expand mask to match feature dimensions
    binary_mask = binary_mask.expand_as(l0)
    
    # Apply Cut-MixMo formula: 2[1_M ⊙ l0 + (1 - 1_M) ⊙ l1]
    mixed_features = 2 * (binary_mask * l0 + (1 - binary_mask) * l1)
    
    return mixed_features, kappa, masks
```

Using CutMixMo in a multi-branch network:

```python
class MixMoNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a backbone with shared parameters
        self.backbone = create_backbone()
        
        # Projection layer for final output
        self.projector = nn.Linear(512, num_classes)
    
    def forward(self, x1, x2):
        # Process the two inputs through the backbone
        feat1 = self.backbone(x1)
        feat2 = self.backbone(x2)
        
        # Apply CutMixMo to the features
        mixed_features, _, _ = cut_mixmo(feat1, feat2, alpha=2.0)
        
        # Apply the final projection
        output = self.projector(mixed_features.flatten(1))
        
        return output
```

## Example: Visualizing CutMix

Here's how to visualize the CutMix augmentation:

```python
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import numpy as np

# Setup data
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# Get a batch of images
images, labels = next(iter(dataloader))

# Initialize CutMix
cutmix = v2.CutMix(num_classes=10, alpha=1.0)

# Apply CutMix
mixed_images, mixed_labels = cutmix(images, labels)

# Convert to numpy for visualization
def show_images(images, title):
    plt.figure(figsize=(12, 8))
    for i in range(len(images)):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        plt.title(f"Index {i}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Show original images
show_images(images, "Original Images")

# Show CutMix images
show_images(mixed_images, "CutMix Images")

# Print the mixed labels (one-hot encoded)
print("Mixed Labels (One-Hot):", mixed_labels)
```

## Using WideResNet and PreActResNet with Feature-Level Augmentation

The repository provides implementations of WideResNet28 and PreActResNet18 with feature-level augmentation support.

### Creating and Using WideResNet28

```python
from models.wide_resnet import Wide_ResNet28
import torch

# Create a WideResNet28 model with CutMixMo augmentation
model = Wide_ResNet28(
    widen_factor=10,           # Controls network width
    dropout_rate=0.3,          # Dropout rate for regularization
    num_classes=10,            # Number of output classes (e.g., for CIFAR-10)
    augmentation_type='CutMixMo'  # Options: 'none', 'LinearMixMo', 'CutMixMo'
)

# Example forward pass with two input batches
batch_size = 32
x1 = torch.randn(batch_size, 3, 32, 32)  # First input batch
x2 = torch.randn(batch_size, 3, 32, 32)  # Second input batch

# Forward pass with feature augmentation
out1, out2, out_mix1, out_mix2, kappa = model(x1, x2)

# Use the outputs in your loss function
# out1, out2: Original outputs from each branch
# out_mix1, out_mix2: Outputs from mixed features
# kappa: Mixing coefficients (useful for some loss functions)
```

### Creating and Using PreActResNet18

```python
from models.preact_resnet import PreActResNet18
import torch

# Create a PreActResNet18 model with LinearMixMo augmentation
model = PreActResNet18(
    widen_factor=2,               # Controls network width
    num_classes=10,               # Number of output classes
    augmentation_type='LinearMixMo'  # Options: 'none', 'LinearMixMo', 'CutMixMo'
)

# Example forward pass
batch_size = 32
x1 = torch.randn(batch_size, 3, 32, 32)
x2 = torch.randn(batch_size, 3, 32, 32)

# Forward pass with feature augmentation
out1, out2, out_mix1, out_mix2, lam = model(x1, x2)

# lam is the mixing coefficient for LinearMixMo
```

### Training a MixMo Model

When training with MixMo augmentation, you'll need to handle the dual inputs and multiple outputs:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from data_handler import DataHandler
from models.wide_resnet import Wide_ResNet28

# Create model
model = Wide_ResNet28(widen_factor=10, dropout_rate=0.3, num_classes=10, 
                     augmentation_type='CutMixMo')

# Setup optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# Get data with batch repetition (each unique sample appears twice in a batch)
data_handler = DataHandler(data_root='./data')
train_loader, _ = data_handler.get_cifar10(batch_size=128, batch_repetitions=2)

# Training loop
model.train()
for inputs, targets in train_loader:
    # Reshape the inputs and targets to separate the batch repetitions
    batch_size = inputs.size(0) // 2
    x1, x2 = inputs[:batch_size], inputs[batch_size:]
    y1, y2 = targets[:batch_size], targets[batch_size:]
    
    # Zero the parameter gradients
    optimizer.zero_grad()
    
    # Forward pass
    out1, out2, out_mix1, out_mix2, kappa = model(x1, x2)
    
    # Calculate loss
    # 1. Original outputs loss
    loss1 = criterion(out1, y1)
    loss2 = criterion(out2, y2)
    
    # 2. Mixed outputs loss
    loss_mix1 = criterion(out_mix1, y1)
    loss_mix2 = criterion(out_mix2, y2)
    
    # Combine losses
    loss = loss1 + loss2 + loss_mix1 + loss_mix2
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()

```


## References

- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899)
- [MixMo: Mixing Multiple Inputs for Multiple Outputs via Deep Subnetworks](https://arxiv.org/abs/2103.06132)
- [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781)