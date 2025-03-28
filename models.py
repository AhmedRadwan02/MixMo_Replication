import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from MixMo import cut_mixmo, linear_mixmo

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )
    
    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class Wide_ResNet_Encoder(nn.Module):
    """Encoder part of Wide ResNet"""
    def __init__(self, depth, widen_factor, dropout_rate):
        super(Wide_ResNet_Encoder, self).__init__()
        self.in_planes = 16
        assert ((depth-4)%6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor
        
        nStages = [16, 16*k, 32*k, 64*k]
        
        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        
        # No custom weight initialization
    
    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Extracts features from the input"""
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        return out
        
    def get_embedding(self, x):
        """Forward pass until the spatial feature map (before pooling)"""
        out = self.forward(x)
        return out

class Wide_ResNet_Dual(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, augmentation_type='none'):
        super(Wide_ResNet_Dual, self).__init__()
        k = widen_factor
        print(f'| Wide-ResNet {depth}x{k} with dual encoders using {augmentation_type} augmentation')
        
        # Create two encoders with same architecture
        self.encoder1 = Wide_ResNet_Encoder(depth, widen_factor, dropout_rate)
        self.encoder2 = Wide_ResNet_Encoder(depth, widen_factor, dropout_rate)
        
        # Store feature dimensions
        self.feature_dim = 64 * widen_factor  # This is the channel dimension after layer3
        
        # Core network (for processing mixed features)
        # For WideResNet, we'll use a simple pooling approach
        
        # Classifiers for each branch
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
        
        # Store augmentation type
        self.augmentation_type = augmentation_type
    
    def set_mixing_mode(self, mode):
        """Set the mixing mode to be used in forward pass
        Args:
            mode: str, either 'patch' or 'linear'
        """
        self.mixing_mode = mode
    
    def forward(self, x1, x2=None):
        """Forward pass through the network with two separate inputs
        
        In inference mode, x2 can be None, in which case x1 is used for both branches
        """
        # Handle inference case (when x2 is None)
        if x2 is None:
            x2 = x1
        
        # Extract features using the two encoders
        features1 = self.encoder1.get_embedding(x1)
        features2 = self.encoder2.get_embedding(x2)
        
        # Original outputs (without mixing)
        out1 = self.classifier1(features1)
        out2 = self.classifier2(features2)
        
        # Apply feature augmentation based on the specified type
        if self.augmentation_type.lower() == 'linearmixmo':
            # Linear interpolation between feature maps
            mixed_features, kappa = linear_mixmo(features1, features2, alpha=2.0)
            out_mix1 = self.classifier1(mixed_features)
            out_mix2 = self.classifier2(mixed_features)
            return out1, out2, out_mix1, out_mix2, kappa
            
        elif self.augmentation_type.lower() == 'cutmixmo':
            # Check if we should use patch mixing or linear mixing
            # Default to patch mixing if mixing_mode is not set
            use_patch_mixing = True
            if hasattr(self, 'mixing_mode'):
                use_patch_mixing = (self.mixing_mode == 'patch')
            
            if use_patch_mixing:
                # Apply CutMixMo on the feature maps (patch mixing)
                mixed_features, kappa = cut_mixmo(features1, features2, alpha=2.0)
            else:
                # Use linear mixing instead (accommodate for inference mode)
                mixed_features, kappa = linear_mixmo(features1, features2, alpha=2.0)
            
            out_mix1 = self.classifier1(mixed_features)
            out_mix2 = self.classifier2(mixed_features)
            return out1, out2, out_mix1, out_mix2, kappa
            
        else:  # No augmentation or unknown type
            # For no augmentation, just return the two original outputs
            # and placeholder None values for compatibility
            return out1, out2, None, None, None

def Wide_ResNet28(widen_factor=10, dropout_rate=0.3, 
                  num_classes=10, augmentation_type='none'):
    """Factory function to create a Wide ResNet 28 with specified augmentation"""
    return Wide_ResNet_Dual(28, widen_factor, dropout_rate, 
                           num_classes, augmentation_type)


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock for PreActResNet"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, widen_factor=1):
        super(PreActBlock, self).__init__()
        planes = planes * widen_factor  # Apply widening factor
        
        # Pre-activation layers
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # Shortcut connection
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet_Encoder(nn.Module):
    """Encoder part of PreActResNet with variable width"""
    def __init__(self, block, num_blocks, widen_factor=1, in_channels=3):
        super(PreActResNet_Encoder, self).__init__()
        self.in_planes = 64
        self.widen_factor = widen_factor
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # ResNet blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Save the feature dimension for the main model
        self.feature_dim = 512 * block.expansion * widen_factor

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.widen_factor))
            self.in_planes = planes * block.expansion * self.widen_factor
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass until the final layer"""
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
        
    def get_embedding(self, x):
        """Forward pass until the spatial feature map (before pooling)"""
        out = self.forward(x)
        return out

class PreActResNet_Dual(nn.Module):
    """PreActResNet with dual encoders and feature mixing"""
    def __init__(self, block, num_blocks, widen_factor=1, num_classes=10, augmentation_type='none'):
        super(PreActResNet_Dual, self).__init__()
        print(f'| PreActResNet-{sum(num_blocks)*2+2}-w{widen_factor} with dual encoders using {augmentation_type} augmentation')
        
        # Create two encoders with same architecture
        self.encoder1 = PreActResNet_Encoder(block, num_blocks, widen_factor)
        self.encoder2 = PreActResNet_Encoder(block, num_blocks, widen_factor)
        
        # Classifiers for each branch
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder1.feature_dim, num_classes)
        )
        
        self.classifier2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder2.feature_dim, num_classes)
        )
        
        # Store augmentation type
        self.augmentation_type = augmentation_type
    
    def forward(self, x1, x2=None):
        """Forward pass through the network with two separate inputs
        
        In inference mode, x2 can be None, in which case x1 is used for both branches
        """
        # Handle inference case (when x2 is None)
        if x2 is None:
            x2 = x1
        
        # Extract features using the two encoders
        features1 = self.encoder1.get_embedding(x1)
        features2 = self.encoder2.get_embedding(x2)
        
        # Original outputs (without mixing)
        out1 = self.classifier1(features1)
        out2 = self.classifier2(features2)
        
        # Apply feature augmentation based on the specified type
        if self.augmentation_type.lower() == 'linearmixmo':
            # Linear interpolation between feature maps
            mixed_features, kappa = linear_mixmo(features1, features2, alpha=2.0)
            out_mix1 = self.classifier1(mixed_features)
            out_mix2 = self.classifier2(mixed_features)
            return out1, out2, out_mix1, out_mix2, kappa
            
        elif self.augmentation_type.lower() == 'cutmixmo':
            # Check if we should use patch mixing or linear mixing
            # Default to patch mixing if mixing_mode is not set
            use_patch_mixing = True
            if hasattr(self, 'mixing_mode'):
                use_patch_mixing = (self.mixing_mode == 'patch')
            
            if use_patch_mixing:
                # Apply CutMixMo on the feature maps (patch mixing)
                mixed_features, kappa = cut_mixmo(features1, features2, alpha=2.0)
            else:
                # Use linear mixing instead (accommodate for inference mode)
                mixed_features, kappa = linear_mixmo(features1, features2, alpha=2.0)
            
            out_mix1 = self.classifier1(mixed_features)
            out_mix2 = self.classifier2(mixed_features)
            return out1, out2, out_mix1, out_mix2, kappa
            
        else:  # No augmentation or unknown type
            # For no augmentation, just return the two original outputs
            # and placeholder None values for compatibility
            return out1, out2, None, None, None

def PreActResNet18(widen_factor=2, num_classes=10, augmentation_type='none'):
    """Create a PreActResNet-18-w with dual encoders and specified augmentation"""
    return PreActResNet_Dual(PreActBlock, [2, 2, 2, 2], widen_factor, num_classes, augmentation_type)

