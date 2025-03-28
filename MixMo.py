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
    
    Where 1_M is a binary mask with area ratio κ ~ Beta(α, α),
    valued at 1 either on a rectangle or on the complementary of a rectangle.
    
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
        # Calculate rectangle dimensions for the given area ratio kappa
        # Ensure dimensions are at least 1 to avoid division by zero errors
        
        # Step 1: Calculate total area to be covered by the mask
        target_area = kappa[i].item() * height * width
        
        # Step 2: Calculate rectangle dimensions (approximating a square where possible)
        # We're calculating sqrt(target_area) and then adjusting as needed
        rect_h = int(math.sqrt(target_area))
        if rect_h < 1:  # Ensure at least 1 pixel height
            rect_h = 1
        
        # Calculate width based on height to match the target area
        rect_w = int(target_area / rect_h)
        if rect_w < 1:  # Ensure at least 1 pixel width
            rect_w = 1
            
        # Adjust dimensions to avoid exceeding the feature map
        rect_h = min(rect_h, height)
        rect_w = min(rect_w, width)
        
        # Step 3: Sample random location for the rectangle
        # Ensure we don't go out of bounds
        y = 0
        x = 0
        
        if height > rect_h:
            y = torch.randint(0, height - rect_h + 1, (1,)).item()
        if width > rect_w:
            x = torch.randint(0, width - rect_w + 1, (1,)).item()
        
        # Step 4: Create the binary mask
        mask = torch.zeros(height, width, device=device)
        mask[y:y+rect_h, x:x+rect_w] = 1
        
        # Step 5: Randomly decide whether to use the mask or its complement
        # This matches the paper's description that the mask is valued at 1
        # either on a rectangle or on the complementary of a rectangle
        if torch.rand(1).item() > 0.5:
            mask = 1 - mask
        
        masks.append(mask)
    
    # Stack masks and add channel dimension for broadcasting
    binary_mask = torch.stack(masks)[:, None, :, :]
    
    # Expand mask to match feature dimensions (across all channels)
    binary_mask = binary_mask.expand_as(l0)
    
    # Apply Cut-MixMo formula: 2[1_M ⊙ l0 + (1 - 1_M) ⊙ l1]
    # The factor of 2 is explicitly mentioned in the paper
    mixed_features = 2 * (binary_mask * l0 + (1 - binary_mask) * l1)
    
    # Return the mixed features and the mixing ratio kappa
    return mixed_features, kappa


    
def linear_mixmo(l0, l1, alpha=2.0):
    """
    Implementation of Linear-MixMo augmentation from the paper:
    "MixMo: Mixing Multiple Inputs for Multiple Outputs via Deep Subnetworks"
    
    This implements the mixing block from section 3.2 of the paper:
    MLinear-MixMo(l0, l1) = 2[κl0 + (1-κ)l1]
    
    Where κ ~ Beta(α, α).
    
    Args:
        l0: First feature embedding tensor [batch_size, channels, height, width]
        l1: Second feature embedding tensor [batch_size, channels, height, width]
        alpha: Parameter for Beta distribution (default 2.0)
        
    Returns:
        Mixed feature embeddings and the mixing ratios κ used
    """
    # Get tensor dimensions
    batch_size = l0.shape[0]
    device = l0.device
    
    # Sample mixing ratio kappa from Beta(α, α)
    kappa = torch.distributions.Beta(alpha, alpha).sample((batch_size,)).to(device)
    
    # Reshape kappa for proper broadcasting
    kappa_reshaped = kappa.view(batch_size, 1, 1, 1)
    
    # Apply Linear-MixMo formula: 2[κl0 + (1-κ)l1]
    # The factor of 2 is explicitly mentioned in the paper
    mixed_features = 2 * (kappa_reshaped * l0 + (1 - kappa_reshaped) * l1)
    
    # Return the mixed features and the mixing ratio kappa
    return mixed_features, kappa
