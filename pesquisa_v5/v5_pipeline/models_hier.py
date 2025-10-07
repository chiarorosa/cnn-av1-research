"""Model definitions for the hierarchical AV1 CNN pipeline (v5).

This module provides a shared lightweight backbone plus task-specific
heads for the binary stage-1 classifier, the macro-class stage-2
classifier, and optional specialist heads for stage-3 refinements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution used throughout the backbone."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Backbone + heads
# ---------------------------------------------------------------------------


class HierarchicalBackbone(nn.Module):
    """Lightweight encoder shared across all stages."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 4]
        self.stem = ConvStem(in_channels, widths[0])
        self.blocks = nn.ModuleList()
        in_c = widths[0]
        for idx, out_c in enumerate(widths[1:], start=1):
            stride = 2 if idx < len(widths) - 1 else 1
            self.blocks.append(DepthwiseSeparableConv(in_c, out_c, stride=stride))
            in_c = out_c
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)


class QPEmbedding(nn.Module):
    """Small linear projection for scalar QP inputs."""

    def __init__(self, embed_dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, qp: Optional[torch.Tensor]) -> torch.Tensor:
        if qp is None:
            return torch.zeros(0)
        if qp.dim() == 1:
            qp = qp.unsqueeze(-1)
        return self.proj(qp)


class Stage1Head(nn.Module):
    def __init__(self, feature_dim: int, qp_dim: int = 0) -> None:
        super().__init__()
        hidden = feature_dim // 2
        input_dim = feature_dim + qp_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features).squeeze(-1)


class Stage2Head(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, qp_dim: int = 0) -> None:
        super().__init__()
        hidden = feature_dim // 2
        input_dim = feature_dim + qp_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class SpecialistHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, qp_dim: int = 0) -> None:
        super().__init__()
        input_dim = feature_dim + qp_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, feature_dim // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


@dataclass
class HierarchicalOutputs:
    stage1: torch.Tensor
    stage2: Optional[torch.Tensor]
    specialists: Dict[str, torch.Tensor]


class HierarchicalModel(nn.Module):
    """Combines backbone, optional QP embedding, and stage-specific heads."""

    def __init__(
        self,
        feature_dim: int,
        stage2_classes: int,
        specialist_classes: Dict[str, int],
        use_qp: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = HierarchicalBackbone()
        self.feature_dim = feature_dim
        self.use_qp = use_qp
        self.qp_embed = QPEmbedding(embed_dim=16) if use_qp else None
        qp_dim = 16 if use_qp else 0

        self.stage1_head = Stage1Head(feature_dim, qp_dim)
        self.stage2_head = Stage2Head(feature_dim, stage2_classes, qp_dim)
        self.specialist_heads = nn.ModuleDict({
            name: SpecialistHead(feature_dim, num_classes, qp_dim)
            for name, num_classes in specialist_classes.items()
        })

    def forward(
        self,
        image: torch.Tensor,
        qp: Optional[torch.Tensor] = None,
    ) -> HierarchicalOutputs:
        features = self.backbone(image)
        if self.use_qp:
            if qp is None:
                qp_embed = torch.zeros(features.size(0), 16, device=features.device)
            else:
                qp_embed = self.qp_embed(qp)
                if qp_embed.dim() == 1:
                    qp_embed = qp_embed.unsqueeze(0)
            features = torch.cat([features, qp_embed], dim=-1)

        stage1_logits = self.stage1_head(features)
        stage2_logits = self.stage2_head(features)
        specialist_logits = {
            name: head(features) for name, head in self.specialist_heads.items()
        }
        return HierarchicalOutputs(
            stage1=stage1_logits,
            stage2=stage2_logits,
            specialists=specialist_logits,
        )


def build_hierarchical_model(
    stage2_classes: int,
    specialist_classes: Dict[str, int],
    use_qp: bool = False,
) -> HierarchicalModel:
    dummy = HierarchicalBackbone()
    feature_dim = dummy(torch.zeros(1, 1, 16, 16)).shape[-1]
    model = HierarchicalModel(
        feature_dim=feature_dim,
        stage2_classes=stage2_classes,
        specialist_classes=specialist_classes,
        use_qp=use_qp,
    )
    return model


__all__ = [
    "HierarchicalBackbone",
    "HierarchicalModel",
    "HierarchicalOutputs",
    "Stage1Head",
    "build_hierarchical_model",
]
