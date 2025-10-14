"""
V7 Pipeline - Backbone Architecture
ResNet-18 + SE-Block + Spatial Attention

Refactored from v6 - contains only essential backbone components.

References:
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Hu, J., et al. (2018). Squeeze-and-Excitation Networks. CVPR.
- Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. ECCV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Hu et al., 2018)
    Adaptively recalibrates channel-wise feature responses
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (Woo et al., 2018)
    Part of CBAM - focuses on 'where' important features are
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)


class ImprovedBackbone(nn.Module):
    """
    ResNet-18 (He et al., 2016) with SE-Blocks and Spatial Attention
    
    Output: 512-dimensional feature vector
    
    Args:
        pretrained (bool): Use ImageNet pre-trained weights
        return_intermediate (bool): Return intermediate layer features for adapters
    """
    def __init__(self, pretrained=True, return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        
        # Base ResNet-18 (He et al., 2016)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Replace first conv to accept 1 channel (grayscale YUV)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                # Initialize with average of RGB channels
                self.conv1.weight[:] = resnet.conv1.weight.mean(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Residual layers
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # SE-Blocks (Hu et al., 2018) - channel attention
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)
        
        # Spatial Attention (Woo et al., 2018)
        self.spatial_attn = SpatialAttention()
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 1, H, W] (grayscale YUV blocks)
        
        Returns:
            If return_intermediate=False: [B, 512] feature vector
            If return_intermediate=True: dict with intermediate features
        """
        intermediates = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.se1(x)
        if self.return_intermediate:
            intermediates['layer1'] = x
        
        x = self.layer2(x)
        x = self.se2(x)
        if self.return_intermediate:
            intermediates['layer2'] = x
        
        x = self.layer3(x)
        x = self.se3(x)
        if self.return_intermediate:
            intermediates['layer3'] = x
        
        x = self.layer4(x)
        x = self.se4(x)
        x = self.spatial_attn(x)
        if self.return_intermediate:
            intermediates['layer4'] = x
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if self.return_intermediate:
            intermediates['features'] = x
            return x, intermediates
        return x


class ClassificationHead(nn.Module):
    """
    Generic classification head for hierarchical stages
    
    Args:
        in_features: Input feature dimension (512 from backbone)
        num_classes: Number of output classes
        dropout: Dropout rate
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(self, in_features=512, num_classes=2, dropout=0.3, hidden_dims=[256]):
        super().__init__()
        
        layers = []
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.head = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.head(x)


# Convenience factory functions for each stage
def create_stage1_head(**kwargs):
    """Stage 1: Binary (NONE vs PARTITION)"""
    return ClassificationHead(num_classes=1, hidden_dims=[256], dropout=0.3, **kwargs)

def create_stage2_head(**kwargs):
    """Stage 2: 3-way (SPLIT, RECT, AB)"""
    return ClassificationHead(num_classes=3, hidden_dims=[256, 128], dropout=0.4, **kwargs)

def create_stage3_rect_head(**kwargs):
    """Stage 3-RECT: Binary (HORZ vs VERT)"""
    return ClassificationHead(num_classes=2, hidden_dims=[128, 64], dropout=0.2, **kwargs)

def create_stage3_ab_head(**kwargs):
    """Stage 3-AB: 4-way (HORZ_A, HORZ_B, VERT_A, VERT_B)"""
    return ClassificationHead(num_classes=4, hidden_dims=[256, 128], dropout=0.5, **kwargs)
