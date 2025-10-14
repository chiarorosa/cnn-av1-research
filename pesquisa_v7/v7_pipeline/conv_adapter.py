"""
V7 Pipeline - Conv-Adapter Module
Parameter Efficient Transfer Learning for ConvNets

Based on:
Chen, H., et al. (2024). Conv-Adapter: Exploring Parameter Efficient Transfer Learning 
for ConvNets. CVPR Workshop.

Key concepts:
1. Freeze pre-trained backbone after Stage 1
2. Learn task-specific feature modulation: h ← h + α·Δh
3. Bottleneck structure with depth-wise separable convolutions
4. Only 3-5% of full fine-tuning parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAdapter(nn.Module):
    """
    Conv-Adapter module for parameter-efficient transfer learning
    
    Architecture (Chen et al., 2024):
    - Down-projection: point-wise conv (channel reduction)
    - Depth-wise conv: maintains spatial locality
    - Up-projection: point-wise conv (channel expansion)
    - Learnable scaling: α parameter
    
    Args:
        in_channels: Input feature channels
        reduction: Channel reduction factor (γ in paper)
        kernel_size: Kernel size for depth-wise conv
        variant: Adapter variant ('conv_parallel', 'conv_sequential', etc.)
    """
    def __init__(self, in_channels, reduction=4, kernel_size=3, variant='conv_parallel'):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.variant = variant
        
        hidden_channels = in_channels // reduction
        padding = kernel_size // 2
        
        # Down-projection (point-wise)
        self.down_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        
        # Depth-wise convolution (maintains locality)
        self.dw_conv = nn.Conv2d(
            hidden_channels, hidden_channels, 
            kernel_size=kernel_size, 
            padding=padding,
            groups=hidden_channels,  # Depth-wise
            bias=False
        )
        
        # Up-projection (point-wise)
        self.up_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        
        # Non-linearity
        self.activation = nn.ReLU(inplace=True)
        
        # Learnable scaling vector α (Chen et al., 2024, Eq. 1)
        # Initialized to ones (identity at start)
        self.alpha = nn.Parameter(torch.ones(in_channels))
        
        # Batch normalization for stability
        self.bn = nn.BatchNorm2d(hidden_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize near-identity to preserve pre-trained features"""
        # Down and up projections: small random weights
        nn.init.kaiming_normal_(self.down_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dw_conv.weight, mode='fan_out', nonlinearity='relu')
        
        # Scale weights down to start near identity
        with torch.no_grad():
            self.down_proj.weight *= 0.01
            self.up_proj.weight *= 0.01
            self.dw_conv.weight *= 0.01
    
    def forward(self, h):
        """
        Feature modulation: h ← h + α·Δh (Chen et al., 2024, Eq. 1)
        
        Args:
            h: Input features [B, C, H, W] from frozen backbone
        
        Returns:
            Modulated features [B, C, H, W]
        """
        # Compute Δh (task-specific modulation)
        delta_h = self.down_proj(h)
        delta_h = self.bn(delta_h)
        delta_h = self.activation(delta_h)
        delta_h = self.dw_conv(delta_h)
        delta_h = self.activation(delta_h)
        delta_h = self.up_proj(delta_h)
        
        # Apply learnable scaling α
        # Reshape α: [C] → [1, C, 1, 1] for broadcasting
        alpha = self.alpha.view(1, -1, 1, 1)
        
        # Residual connection: h + α·Δh
        return h + alpha * delta_h
    
    def num_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdapterBackbone(nn.Module):
    """
    Backbone with Conv-Adapters inserted after specific layers
    
    Adapting schemes (Chen et al., 2024, Fig. 3):
    - 'conv_parallel': Adapt K×K conv layers in parallel
    - 'conv_sequential': Adapt K×K conv layers in sequence
    - 'residual_parallel': Adapt whole residual blocks in parallel
    - 'residual_sequential': Adapt whole residual blocks in sequence
    
    Args:
        backbone: Pre-trained backbone (ImprovedBackbone)
        adapter_config: Dict with adapter settings
            - reduction: Channel reduction factor (default: 4)
            - layers: Which layers to adapt (e.g., ['layer3', 'layer4'])
            - variant: Adapter variant
    """
    def __init__(self, backbone, adapter_config=None):
        super().__init__()
        self.backbone = backbone
        
        # Default config
        if adapter_config is None:
            adapter_config = {
                'reduction': 4,
                'layers': ['layer3', 'layer4'],  # Adapt deep layers
                'variant': 'conv_parallel'
            }
        
        self.config = adapter_config
        
        # Freeze backbone (Chen et al., 2024: pre-trained parameters frozen)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Create adapters for specified layers
        self.adapters = nn.ModuleDict()
        
        layer_channels = {
            'layer1': 64,
            'layer2': 128,
            'layer3': 256,
            'layer4': 512
        }
        
        for layer_name in adapter_config['layers']:
            if layer_name in layer_channels:
                self.adapters[layer_name] = ConvAdapter(
                    in_channels=layer_channels[layer_name],
                    reduction=adapter_config['reduction'],
                    variant=adapter_config['variant']
                )
    
    def forward(self, x):
        """
        Forward pass with adapter modulation
        
        Returns:
            features: [B, 512] final feature vector
            intermediates: Dict with layer outputs (for analysis)
        """
        intermediates = {}
        
        # Initial layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # Layer 1
        x = self.backbone.layer1(x)
        x = self.backbone.se1(x)
        if 'layer1' in self.adapters:
            x = self.adapters['layer1'](x)
            intermediates['adapter1'] = x
        
        # Layer 2
        x = self.backbone.layer2(x)
        x = self.backbone.se2(x)
        if 'layer2' in self.adapters:
            x = self.adapters['layer2'](x)
            intermediates['adapter2'] = x
        
        # Layer 3
        x = self.backbone.layer3(x)
        x = self.backbone.se3(x)
        if 'layer3' in self.adapters:
            x = self.adapters['layer3'](x)
            intermediates['adapter3'] = x
        
        # Layer 4
        x = self.backbone.layer4(x)
        x = self.backbone.se4(x)
        x = self.backbone.spatial_attn(x)
        if 'layer4' in self.adapters:
            x = self.adapters['layer4'](x)
            intermediates['adapter4'] = x
        
        # Global pooling
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        
        return features, intermediates
    
    def num_adapter_parameters(self):
        """Count adapter parameters (should be ~3-5% of full model)"""
        adapter_params = sum(
            adapter.num_parameters() 
            for adapter in self.adapters.values()
        )
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'adapter_params': adapter_params,
            'total_params': total_params,
            'percentage': 100 * adapter_params / total_params
        }


# Factory function for creating adapter-based models
def create_adapter_model(backbone, stage, adapter_config=None):
    """
    Create a full model with adapters for a specific stage
    
    Args:
        backbone: Pre-trained ImprovedBackbone
        stage: Stage number (1, 2, 3)
        adapter_config: Adapter configuration
    
    Returns:
        model: nn.Module with AdapterBackbone + ClassificationHead
    """
    from .backbone import (
        create_stage1_head, create_stage2_head, 
        create_stage3_rect_head, create_stage3_ab_head
    )
    
    adapter_backbone = AdapterBackbone(backbone, adapter_config)
    
    # Select appropriate head
    if stage == 1:
        head = create_stage1_head()
    elif stage == 2:
        head = create_stage2_head()
    elif stage == '3_rect':
        head = create_stage3_rect_head()
    elif stage == '3_ab':
        head = create_stage3_ab_head()
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    class AdapterModel(nn.Module):
        def __init__(self, adapter_backbone, head):
            super().__init__()
            self.adapter_backbone = adapter_backbone
            self.head = head
        
        def forward(self, x):
            features, intermediates = self.adapter_backbone(x)
            logits = self.head(features)
            return logits
        
        def num_adapter_parameters(self):
            return self.adapter_backbone.num_adapter_parameters()
    
    return AdapterModel(adapter_backbone, head)
