import torch
import torch.nn as nn
from typing import Dict, Tuple

from mixmo_base import MixMoBase


class LinearMixMo(MixMoBase):
    """
    Implementation of LinearMixMo - a variant of MixMo that uses linear mixing of inputs.
    
    As described in "MixMo: Mixing Multiple Inputs for Multiple Outputs via Deep Subnetworks"
    by Alexandre Ramé, Rémy Sun, and Matthieu Cord.
    
    This implementation supports M=2 subnetworks.
    """
    
    def mix_features(self, l0: torch.Tensor, l1: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
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


# Example of a simple backbone network to use with LinearMixMo
class SimpleBackbone(nn.Module):
    """
    A simple backbone network for use with LinearMixMo.
    """
    
    def __init__(self, in_channels: int, out_features: int):
        """
        Initialize the backbone.
        
        Args:
            in_channels: Number of input channels
            out_features: Number of output features
        """
        super(SimpleBackbone, self).__init__()
        
        self.out_features = out_features
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the backbone.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
