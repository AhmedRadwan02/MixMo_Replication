import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
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


    
def linear_mixmo(self, l0: torch.Tensor, l1: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """
    Mix features using linear interpolation.
    
    Formula: MLinear-MixMo (l0, l1) = 2[λl0 + (1-λ)l1]
    
    Args:
        l0: Features from first encoder
        l1: Features from second encoder
        lam: Mixing ratios
        
    Returns:
        Mixed features
    """
    # Apply linear mixing (MLinear-MixMo)
    # Formula: MLinear-MixMo (l0, l1) = 2[λl0 + (1-λ)l1]
    mixed_features = 2 * (lam * l0 + (1 - lam) * l1)
    
    return mixed_features
