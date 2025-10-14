"""
V7 Pipeline - Hybrid Model (Conv-Adapter + Ensemble)
Combines parameter-efficient adapters with ensemble voting

Innovation: PhD-level contribution
- Conv-Adapter (Chen et al., CVPR 2024) for negative transfer prevention
- Ensemble voting (Ahad et al., 2024) for error compensation
- Hierarchical application across all stages

Expected benefits:
1. Parameter efficiency: ~10% trainable params (vs 300% for 3 full backbones)
2. Negative transfer prevention: Frozen backbone after Stage 1
3. Ensemble boosting: +1-5% F1 from voting
4. Few-shot robustness: Adapters excel on rare classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from .conv_adapter import ConvAdapter, AdapterBackbone
from .ensemble import SoftVotingEnsemble


class AdapterEnsembleModel(nn.Module):
    """
    Single stage model with ensemble of Conv-Adapters
    
    Architecture:
    1. Shared frozen backbone (trained in Stage 1)
    2. Multiple adapters with different configurations
    3. Soft voting over adapter outputs
    
    Args:
        backbone: Pre-trained ImprovedBackbone (will be frozen)
        num_adapters: Number of adapter variants (default: 3)
        adapter_configs: List of configs for each adapter
        head_factory: Function to create classification head
        voting_learnable: Learn ensemble weights
    """
    def __init__(
        self, 
        backbone,
        num_adapters=3,
        adapter_configs=None,
        head_factory=None,
        voting_learnable=True
    ):
        super().__init__()
        
        # Freeze shared backbone
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Default adapter configurations (diverse settings)
        if adapter_configs is None:
            adapter_configs = [
                {'reduction': 4, 'layers': ['layer3', 'layer4'], 'variant': 'conv_parallel'},
                {'reduction': 8, 'layers': ['layer3', 'layer4'], 'variant': 'conv_parallel'},
                {'reduction': 4, 'layers': ['layer4'], 'variant': 'conv_parallel'},
            ]
        
        # Ensure we have exactly num_adapters configs
        adapter_configs = adapter_configs[:num_adapters]
        
        # Create adapters (parameter-efficient, only ~3-5% params each)
        self.adapters = nn.ModuleList()
        for config in adapter_configs:
            adapter_dict = nn.ModuleDict()
            
            layer_channels = {
                'layer1': 64,
                'layer2': 128,
                'layer3': 256,
                'layer4': 512
            }
            
            for layer_name in config['layers']:
                if layer_name in layer_channels:
                    adapter_dict[layer_name] = ConvAdapter(
                        in_channels=layer_channels[layer_name],
                        reduction=config['reduction'],
                        variant=config.get('variant', 'conv_parallel')
                    )
            
            self.adapters.append(adapter_dict)
        
        # Classification heads (one per adapter)
        if head_factory is None:
            from .backbone import create_stage2_head
            head_factory = create_stage2_head
        
        self.heads = nn.ModuleList([head_factory() for _ in range(num_adapters)])
        
        # Ensemble weights (learnable or fixed)
        if voting_learnable:
            self.weights = nn.Parameter(torch.ones(num_adapters) / num_adapters)
        else:
            self.register_buffer('weights', torch.ones(num_adapters) / num_adapters)
        
        self.num_adapters = num_adapters
        self.voting_learnable = voting_learnable
    
    def _forward_with_adapter(self, x, adapter_dict):
        """
        Forward pass through backbone with a specific adapter
        
        Args:
            x: Input [B, 1, H, W]
            adapter_dict: ModuleDict with adapters for specific layers
        
        Returns:
            features: [B, 512]
        """
        # Initial layers
        out = self.backbone.conv1(x)
        out = self.backbone.bn1(out)
        out = self.backbone.relu(out)
        out = self.backbone.maxpool(out)
        
        # Layer 1
        out = self.backbone.layer1(out)
        out = self.backbone.se1(out)
        if 'layer1' in adapter_dict:
            out = adapter_dict['layer1'](out)
        
        # Layer 2
        out = self.backbone.layer2(out)
        out = self.backbone.se2(out)
        if 'layer2' in adapter_dict:
            out = adapter_dict['layer2'](out)
        
        # Layer 3
        out = self.backbone.layer3(out)
        out = self.backbone.se3(out)
        if 'layer3' in adapter_dict:
            out = adapter_dict['layer3'](out)
        
        # Layer 4
        out = self.backbone.layer4(out)
        out = self.backbone.se4(out)
        out = self.backbone.spatial_attn(out)
        if 'layer4' in adapter_dict:
            out = adapter_dict['layer4'](out)
        
        # Global pooling
        out = self.backbone.avgpool(out)
        features = torch.flatten(out, 1)
        
        return features
    
    def forward(self, x, return_individual=False):
        """
        Ensemble forward with soft voting
        
        Args:
            x: Input [B, 1, H, W]
            return_individual: Return individual adapter predictions
        
        Returns:
            ensemble_logits: [B, num_classes]
            individual_logits: [num_adapters, B, num_classes] (if requested)
        """
        all_logits = []
        
        # Forward through each adapter + head
        for adapter_dict, head in zip(self.adapters, self.heads):
            features = self._forward_with_adapter(x, adapter_dict)
            logits = head(features)
            all_logits.append(logits)
        
        # Stack: [num_adapters, B, num_classes]
        all_logits = torch.stack(all_logits, dim=0)
        
        # Soft voting with learned weights
        all_probs = F.softmax(all_logits, dim=-1)
        weights = F.softmax(self.weights, dim=0).view(-1, 1, 1)
        ensemble_probs = (all_probs * weights).sum(dim=0)
        
        # Convert back to logits
        ensemble_logits = torch.log(ensemble_probs + 1e-8)
        
        if return_individual:
            return ensemble_logits, all_logits
        return ensemble_logits
    
    def num_parameters_analysis(self):
        """
        Analyze parameter counts
        
        Returns:
            Dict with parameter breakdown
        """
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        
        adapter_params = sum(
            p.numel() for adapter in self.adapters 
            for p in adapter.parameters() if p.requires_grad
        )
        
        head_params = sum(
            p.numel() for head in self.heads 
            for p in head.parameters() if p.requires_grad
        )
        
        total_trainable = adapter_params + head_params
        
        return {
            'backbone_params': backbone_params,
            'backbone_frozen': True,
            'adapter_params': adapter_params,
            'head_params': head_params,
            'total_trainable': total_trainable,
            'percentage_trainable': 100 * total_trainable / (backbone_params + total_trainable),
            'num_adapters': self.num_adapters
        }


class HybridHierarchicalPipeline(nn.Module):
    """
    Full hierarchical pipeline with Adapter-Ensemble at each stage
    
    Architecture:
    Stage 1: Train normally (establish good features)
    Stage 2: Freeze backbone, use AdapterEnsemble
    Stage 3-RECT: Freeze backbone, use AdapterEnsemble
    Stage 3-AB: Freeze backbone, use AdapterEnsemble
    
    Expected performance:
    - Stage 2 F1: 46% → 65-73% (adapter + ensemble)
    - Stage 3-AB F1: 24.5% → 45-50% (few-shot + ensemble)
    - Pipeline F1: ~70-75%
    """
    def __init__(
        self,
        stage1_model,  # Pre-trained Stage 1 (binary)
        backbone,  # Shared frozen backbone
        stage2_config=None,
        stage3_rect_config=None,
        stage3_ab_config=None
    ):
        super().__init__()
        
        # Stage 1: Pre-trained binary classifier
        self.stage1 = stage1_model
        for param in self.stage1.parameters():
            param.requires_grad = False
        
        # Stage 2: Adapter-Ensemble (3-way: SPLIT, RECT, AB)
        from .backbone import create_stage2_head
        if stage2_config is None:
            stage2_config = {'num_adapters': 3, 'voting_learnable': True}
        
        self.stage2 = AdapterEnsembleModel(
            backbone=backbone,
            head_factory=create_stage2_head,
            **stage2_config
        )
        
        # Stage 3-RECT: Adapter-Ensemble (binary: HORZ vs VERT)
        from .backbone import create_stage3_rect_head
        if stage3_rect_config is None:
            stage3_rect_config = {'num_adapters': 3, 'voting_learnable': True}
        
        self.stage3_rect = AdapterEnsembleModel(
            backbone=backbone,
            head_factory=create_stage3_rect_head,
            **stage3_rect_config
        )
        
        # Stage 3-AB: Adapter-Ensemble (4-way: HORZ_A/B, VERT_A/B)
        from .backbone import create_stage3_ab_head
        if stage3_ab_config is None:
            stage3_ab_config = {'num_adapters': 3, 'voting_learnable': True}
        
        self.stage3_ab = AdapterEnsembleModel(
            backbone=backbone,
            head_factory=create_stage3_ab_head,
            **stage3_ab_config
        )
    
    def forward(self, x, return_intermediates=False):
        """
        Full hierarchical pipeline
        
        Returns:
            final_pred: [B] with partition types 0-9
            intermediates: Dict with stage outputs (if requested)
        """
        batch_size = x.size(0)
        device = x.device
        intermediates = {}
        
        # Stage 1: NONE vs PARTITION
        with torch.no_grad():
            stage1_logits = self.stage1(x)
            stage1_pred = (torch.sigmoid(stage1_logits).squeeze() > 0.5)
        
        intermediates['stage1_pred'] = stage1_pred
        
        final_pred = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # NONE blocks
        none_mask = ~stage1_pred
        final_pred[none_mask] = 0  # PARTITION_NONE
        
        if none_mask.all():
            if return_intermediates:
                return final_pred, intermediates
            return final_pred
        
        # Stage 2: SPLIT, RECT, AB (only for PARTITION blocks)
        partition_x = x[stage1_pred]
        stage2_logits = self.stage2(partition_x)
        stage2_pred = torch.argmax(stage2_logits, dim=1)
        
        # Create full-size stage2_pred tensor
        stage2_pred_full = torch.zeros(batch_size, dtype=torch.long, device=device)
        stage2_pred_full[stage1_pred] = stage2_pred
        intermediates['stage2_pred'] = stage2_pred_full
        
        # SPLIT blocks
        split_mask = stage1_pred.clone()
        split_mask[stage1_pred] = (stage2_pred == 0)
        final_pred[split_mask] = 3  # PARTITION_SPLIT
        
        # Stage 3-RECT: HORZ vs VERT
        rect_mask = stage1_pred.clone()
        rect_mask[stage1_pred] = (stage2_pred == 1)
        
        if rect_mask.any():
            rect_x = x[rect_mask]
            rect_logits = self.stage3_rect(rect_x)
            rect_pred = torch.argmax(rect_logits, dim=1)
            final_pred[rect_mask] = rect_pred + 1  # 1: HORZ, 2: VERT
            
            rect_pred_full = torch.zeros(batch_size, dtype=torch.long, device=device)
            rect_pred_full[rect_mask] = rect_pred
            intermediates['stage3_rect_pred'] = rect_pred_full
        
        # Stage 3-AB: HORZ_A, HORZ_B, VERT_A, VERT_B
        ab_mask = stage1_pred.clone()
        ab_mask[stage1_pred] = (stage2_pred == 2)
        
        if ab_mask.any():
            ab_x = x[ab_mask]
            ab_logits = self.stage3_ab(ab_x)
            ab_pred = torch.argmax(ab_logits, dim=1)
            final_pred[ab_mask] = ab_pred + 4  # 4-7: HORZ_A/B, VERT_A/B
            
            ab_pred_full = torch.zeros(batch_size, dtype=torch.long, device=device)
            ab_pred_full[ab_mask] = ab_pred
            intermediates['stage3_ab_pred'] = ab_pred_full
        
        if return_intermediates:
            return final_pred, intermediates
        return final_pred
    
    def get_stage(self, stage_name):
        """Get specific stage for training"""
        if stage_name == 'stage1':
            return self.stage1
        elif stage_name == 'stage2':
            return self.stage2
        elif stage_name == 'stage3_rect':
            return self.stage3_rect
        elif stage_name == 'stage3_ab':
            return self.stage3_ab
        else:
            raise ValueError(f"Unknown stage: {stage_name}")


# Factory function
def create_hybrid_pipeline(stage1_checkpoint_path, backbone, device='cuda'):
    """
    Create HybridHierarchicalPipeline from Stage 1 checkpoint
    
    Args:
        stage1_checkpoint_path: Path to trained Stage 1 model
        backbone: Shared frozen backbone
        device: 'cuda' or 'cpu'
    
    Returns:
        HybridHierarchicalPipeline ready for Stage 2 training
    """
    from .backbone import create_stage1_head
    
    # Load Stage 1
    stage1_model = nn.Sequential(backbone, create_stage1_head())
    checkpoint = torch.load(stage1_checkpoint_path, map_location=device)
    stage1_model.load_state_dict(checkpoint['model_state_dict'])
    stage1_model.eval()
    
    # Create hybrid pipeline
    pipeline = HybridHierarchicalPipeline(
        stage1_model=stage1_model,
        backbone=backbone
    )
    
    return pipeline.to(device)
