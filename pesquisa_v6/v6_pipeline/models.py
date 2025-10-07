"""
V6 Pipeline - Model Architectures
ResNet-18 + SE-Block + Spatial Attention + Specialized Heads

Architectural components based on literature:
1. ResNet-18 (He et al., 2016) - Deep residual learning
2. SE-Block (Hu et al., 2018) - Squeeze-and-Excitation networks
3. Spatial Attention (Woo et al., 2018) - CBAM attention mechanism
4. Temperature Scaling (Guo et al., 2017) - Calibration for binary head

References:
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Hu, J., et al. (2018). Squeeze-and-Excitation Networks. CVPR.
- Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module. ECCV.
- Guo, C., et al. (2017). On Calibration of Modern Neural Networks. ICML.
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
    Combines residual learning with channel and spatial attention
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Base ResNet-18 (He et al., 2016)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Extract layers
        # Replace first conv to accept 1 channel (grayscale)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize with pre-trained weights (average RGB channels)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight[:] = resnet.conv1.weight.mean(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # SE-Blocks (Hu et al., 2018) - channel attention
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)
        
        # Spatial Attention (Woo et al., 2018) - spatial attention
        self.spatial_attn = SpatialAttention()
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.se1(x)
        
        x = self.layer2(x)
        x = self.se2(x)
        
        x = self.layer3(x)
        x = self.se3(x)
        
        x = self.layer4(x)
        x = self.se4(x)
        x = self.spatial_attn(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


class Stage1BinaryHead(nn.Module):
    """
    Binary classification head for NONE vs PARTITION
    Uses Temperature Scaling (Guo et al., 2017) for calibration
    """
    def __init__(self, in_features=512, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        # Temperature Scaling (Guo et al., 2017)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, x, apply_temp=False):
        logits = self.head(x)
        if apply_temp:
            logits = logits / self.temperature
        return logits


class Stage2ThreeWayHead(nn.Module):
    """3-way classification: SPLIT, RECT, AB"""
    def __init__(self, in_features=512, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # SPLIT, RECT, AB
        )
    
    def forward(self, x):
        return self.head(x)


class Stage3RectHead(nn.Module):
    """Binary classification: HORZ vs VERT"""
    def __init__(self, in_features=512, dropout=0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # HORZ, VERT
        )
    
    def forward(self, x):
        return self.head(x)


class Stage3ABHead(nn.Module):
    """4-way classification: HORZ_A, HORZ_B, VERT_A, VERT_B"""
    def __init__(self, in_features=512, dropout=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 4)  # HORZ_A, HORZ_B, VERT_A, VERT_B
        )
    
    def forward(self, x):
        return self.head(x)


class Stage1Model(nn.Module):
    """Complete Stage 1 model"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = ImprovedBackbone(pretrained)
        self.head = Stage1BinaryHead()
    
    def forward(self, x, apply_temp=False):
        features = self.backbone(x)
        return self.head(features, apply_temp)


class Stage2Model(nn.Module):
    """Complete Stage 2 model"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = ImprovedBackbone(pretrained)
        self.head = Stage2ThreeWayHead()
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class Stage3RectModel(nn.Module):
    """Complete Stage 3-RECT model"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = ImprovedBackbone(pretrained)
        self.head = Stage3RectHead()
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class Stage3ABModel(nn.Module):
    """Complete Stage 3-AB model"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = ImprovedBackbone(pretrained)
        self.head = Stage3ABHead()
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    x = torch.randn(batch_size, 3, 16, 16).to(device)
    
    print("Testing Stage 1 Model...")
    model1 = Stage1Model().to(device)
    out1 = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out1.shape}")
    print(f"Expected: ({batch_size}, 1)\n")
    
    print("Testing Stage 2 Model...")
    model2 = Stage2Model().to(device)
    out2 = model2(x)
    print(f"Output shape: {out2.shape}")
    print(f"Expected: ({batch_size}, 3)\n")
    
    print("Testing Stage 3-RECT Model...")
    model3_rect = Stage3RectModel().to(device)
    out3_rect = model3_rect(x)
    print(f"Output shape: {out3_rect.shape}")
    print(f"Expected: ({batch_size}, 2)\n")
    
    print("Testing Stage 3-AB Model...")
    model3_ab = Stage3ABModel().to(device)
    out3_ab = model3_ab(x)
    print(f"Output shape: {out3_ab.shape}")
    print(f"Expected: ({batch_size}, 4)\n")
    
    print("✅ All models working correctly!")
