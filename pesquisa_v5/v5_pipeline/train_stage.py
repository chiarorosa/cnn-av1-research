"""Training helpers for hierarchical stage models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class Stage1Config:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pos_weight: float = 1.0
    focal_gamma: float = 2.0
    device: str = "cuda"


@dataclass
class Stage1Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass
class Stage2Config:
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    class_weights: Optional[torch.Tensor] = None
    label_smoothing: float = 0.0
    device: str = "cuda"
    num_classes: int = 5


@dataclass
class Stage2Metrics:
    loss: float
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: Dict[str, float]


def _binary_stats(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    preds = preds.int()
    targets = targets.int()
    tp = torch.sum((preds == 1) & (targets == 1)).item()
    tn = torch.sum((preds == 0) & (targets == 0)).item()
    fp = torch.sum((preds == 1) & (targets == 0)).item()
    fn = torch.sum((preds == 0) & (targets == 1)).item()
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _stage1_loss(logits: torch.Tensor, targets: torch.Tensor, cfg: Stage1Config) -> torch.Tensor:
    targets = targets.float()
    pos_weight = torch.tensor(cfg.pos_weight, device=logits.device)
    per_example = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    if cfg.focal_gamma <= 0:
        return per_example.mean()
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    focal_factor = (1 - pt).pow(cfg.focal_gamma)
    return (focal_factor * per_example).mean()


def _stage2_loss(logits: torch.Tensor, targets: torch.Tensor, cfg: Stage2Config) -> torch.Tensor:
    weight = None
    if cfg.class_weights is not None:
        weight = cfg.class_weights.to(logits.device)
    return F.cross_entropy(
        logits,
        targets,
        weight=weight,
        label_smoothing=cfg.label_smoothing,
    )


def _multiclass_stats(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    device = preds.device
    conf = torch.bincount(
        targets * num_classes + preds,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes).to(device=torch.device("cpu"), dtype=torch.float32)

    tp = torch.diag(conf)
    support = conf.sum(dim=1)
    predicted = conf.sum(dim=0)

    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(denom))

    accuracy = tp.sum() / conf.sum().clamp(min=1.0)
    macro_f1 = f1.mean().item()
    weighted_f1 = (f1 * (support / support.sum().clamp(min=1.0))).sum().item()
    per_class = {str(idx): f1[idx].item() for idx in range(num_classes)}

    return {
        "accuracy": accuracy.item(),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class,
    }


def _evaluate_stage1(model: nn.Module, loader: DataLoader, cfg: Stage1Config) -> Stage1Metrics:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(cfg.device)
            targets = batch["label_stage1"].to(cfg.device)
            qp = batch.get("qp")
            qp = qp.to(cfg.device) if qp is not None else None
            logits = model(image=images, qp=qp).stage1
            loss = _stage1_loss(logits, targets, cfg)
            total_loss += loss.item() * images.size(0)
            total_examples += images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).cpu()
            all_preds.append(preds)
            all_targets.append(targets.cpu())
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    stats = _binary_stats(preds, targets)
    return Stage1Metrics(
        loss=total_loss / max(total_examples, 1),
        accuracy=stats["accuracy"],
        precision=stats["precision"],
        recall=stats["recall"],
        f1=stats["f1"],
    )


def train_stage1(model: nn.Module, loaders: Dict[str, DataLoader], cfg: Stage1Config) -> Dict[str, Iterable[Stage1Metrics]]:
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = {"train": [], "val": []}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        total_examples = 0
        epoch_preds = []
        epoch_targets = []
        for batch in loaders["train"]:
            images = batch["image"].to(cfg.device)
            targets = batch["label_stage1"].to(cfg.device)
            qp = batch.get("qp")
            qp = qp.to(cfg.device) if qp is not None else None
            optimizer.zero_grad()
            logits = model(image=images, qp=qp).stage1
            loss = _stage1_loss(logits, targets, cfg)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            total_examples += images.size(0)
            epoch_preds.append((torch.sigmoid(logits.detach()) >= 0.5).cpu())
            epoch_targets.append(targets.detach().cpu())

        train_preds = torch.cat(epoch_preds)
        train_targets = torch.cat(epoch_targets)
        train_stats = _binary_stats(train_preds, train_targets)
        train_metrics = Stage1Metrics(
            loss=epoch_loss / max(total_examples, 1),
            accuracy=train_stats["accuracy"],
            precision=train_stats["precision"],
            recall=train_stats["recall"],
            f1=train_stats["f1"],
        )
        history["train"].append(train_metrics)

        val_metrics = _evaluate_stage1(model, loaders["val"], cfg)
        history["val"].append(val_metrics)

        print(
            f"[Stage1] Epoch {epoch + 1}/{cfg.epochs} | "
            f"Train loss: {train_metrics.loss:.4f}, F1: {train_metrics.f1:.3f} | "
            f"Val loss: {val_metrics.loss:.4f}, F1: {val_metrics.f1:.3f}")

    return history


def _evaluate_stage2(model: nn.Module, loader: DataLoader, cfg: Stage2Config) -> Stage2Metrics:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(cfg.device)
            targets = batch["label_stage2"].to(cfg.device)
            qp = batch.get("qp")
            qp = qp.to(cfg.device) if qp is not None else None
            logits = model(image=images, qp=qp).stage2
            loss = _stage2_loss(logits, targets, cfg)
            total_loss += loss.item() * images.size(0)
            total_examples += images.size(0)
            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(targets.cpu())

    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    stats = _multiclass_stats(preds_cat, targets_cat, cfg.num_classes)
    return Stage2Metrics(
        loss=total_loss / max(total_examples, 1),
        accuracy=stats["accuracy"],
        macro_f1=stats["macro_f1"],
        weighted_f1=stats["weighted_f1"],
        per_class_f1=stats["per_class_f1"],
    )


def train_stage2(model: nn.Module, loaders: Dict[str, DataLoader], cfg: Stage2Config) -> Dict[str, Iterable[Stage2Metrics]]:
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    history = {"train": [], "val": []}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        total_examples = 0
        all_preds = []
        all_targets = []
        for batch in loaders["train"]:
            images = batch["image"].to(cfg.device)
            targets = batch["label_stage2"].to(cfg.device)
            qp = batch.get("qp")
            qp = qp.to(cfg.device) if qp is not None else None
            optimizer.zero_grad()
            logits = model(image=images, qp=qp).stage2
            loss = _stage2_loss(logits, targets, cfg)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            total_examples += images.size(0)
            preds = torch.argmax(logits.detach(), dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(targets.detach().cpu())

        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)
        stats = _multiclass_stats(preds_cat, targets_cat, cfg.num_classes)
        train_metrics = Stage2Metrics(
            loss=epoch_loss / max(total_examples, 1),
            accuracy=stats["accuracy"],
            macro_f1=stats["macro_f1"],
            weighted_f1=stats["weighted_f1"],
            per_class_f1=stats["per_class_f1"],
        )
        history["train"].append(train_metrics)

        val_metrics = _evaluate_stage2(model, loaders["val"], cfg)
        history["val"].append(val_metrics)

        print(
            f"[Stage2] Epoch {epoch + 1}/{cfg.epochs} | "
            f"Train loss: {train_metrics.loss:.4f}, macro-F1: {train_metrics.macro_f1:.3f} | "
            f"Val loss: {val_metrics.loss:.4f}, macro-F1: {val_metrics.macro_f1:.3f}")

    return history


__all__ = [
    "Stage1Config",
    "Stage1Metrics",
    "Stage2Config",
    "Stage2Metrics",
    "train_stage1",
    "train_stage2",
]
