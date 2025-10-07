#!/usr/bin/env python3
"""Treina o especialista Stage3 (RECT, AB ou 1TO4) usando o backbone Stage2."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable

import torch
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pesquisa.v5_pipeline import (  # noqa: E402
    STAGE2_GROUPS,
    STAGE3_GROUPS,
    build_hierarchical_model,
)
import torch.nn as nn


class TensorDictDataset(Dataset):
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.tensors = tensors
        self.length = next(iter(tensors.values())).shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.tensors.items()}


@dataclass
class Stage3Config:
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    label_smoothing: float = 0.0


@dataclass
class Stage3Metrics:
    loss: float
    accuracy: float
    macro_f1: float
    per_class_f1: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=["RECT", "AB", "1TO4"], required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("pesquisa/v5_dataset_stage3"))
    parser.add_argument("--block-size", choices=["8", "16", "32", "64"], default="16")
    parser.add_argument("--stage2-state", type=Path, default=Path("pesquisa/logs/v5_stage2/stage2_state_dict_block16_ep8.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=Path("pesquisa/logs/v5_stage3"))
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--use-sampler", action="store_true")
    return parser.parse_args()


def _class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = torch.where(counts > 0, counts.sum() / (num_classes * counts), torch.zeros_like(counts))
    weights = weights.pow(2.0)
    weights = weights / weights.mean().clamp(min=1e-6)
    return weights


def _metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Stage3Metrics:
    conf = torch.bincount(
        targets * num_classes + preds,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes).float()
    tp = conf.diag()
    support = conf.sum(dim=1)
    predicted = conf.sum(dim=0)
    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(denom))
    macro_f1 = f1.mean().item()
    accuracy = tp.sum() / conf.sum().clamp(min=1.0)
    per_class = {str(i): f1[i].item() for i in range(num_classes)}
    return Stage3Metrics(
        loss=0.0,  # preenchido posteriormente
        accuracy=accuracy.item(),
        macro_f1=macro_f1,
        per_class_f1=per_class,
    )


def _evaluate(model, loader: DataLoader, cfg: Stage3Config, head: str, num_classes: int, class_weights: torch.Tensor) -> Stage3Metrics:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(cfg.device)
            labels = batch["label"].to(cfg.device)
            qp = batch.get("qp")
            qp = qp.to(cfg.device) if qp is not None else None

            outputs = model(image=images, qp=qp)
            logits = outputs.specialists[head]
            loss = F.cross_entropy(logits, labels, weight=class_weights.to(cfg.device), label_smoothing=cfg.label_smoothing)
            total_loss += loss.item() * images.size(0)
            total_examples += images.size(0)
            preds_all.append(torch.argmax(logits, dim=1).cpu())
            targets_all.append(labels.cpu())

    preds = torch.cat(preds_all)
    targets = torch.cat(targets_all)
    metrics = _metrics(preds, targets, num_classes)
    metrics.loss = total_loss / max(total_examples, 1)
    return metrics


def main() -> None:
    args = parse_args()
    head = args.head
    block_name = f"block_{args.block_size}"
    dataset_dir = args.dataset_root / head / block_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset Stage3 não encontrado: {dataset_dir}")

    train_bundle = torch.load(dataset_dir / "train.pt")
    val_bundle = torch.load(dataset_dir / "val.pt")

    train_dataset = TensorDictDataset(train_bundle)
    val_dataset = TensorDictDataset(val_bundle)

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
    }

    num_classes = len(STAGE3_GROUPS[head])
    if args.use_sampler:
        weights = _class_weights(train_bundle["label"], num_classes)
        sampler_weights = weights[train_bundle["label"]]
        train_loader_kwargs["sampler"] = WeightedRandomSampler(sampler_weights, len(sampler_weights), replacement=True)
    else:
        train_loader_kwargs["shuffle"] = True

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_hierarchical_model(
        stage2_classes=len(STAGE2_GROUPS),
        specialist_classes={head: num_classes},
        use_qp=False,
    )

    raw_state = torch.load(args.stage2_state, map_location="cpu")
    state_dict = {
        k: v for k, v in raw_state.items()
        if not k.startswith("stage2_head.") and not k.startswith("specialist_heads.")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Stage3] Missing keys ao carregar checkpoint: {missing}")
    if unexpected:
        print(f"[Stage3] Unexpected keys ao carregar checkpoint: {unexpected}")

    # Reset treinável somente para o especialista alvo
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.stage1_head.parameters():
        param.requires_grad = False
    for param in model.stage2_head.parameters():
        param.requires_grad = False
    for name, head_module in model.specialist_heads.items():
        requires = name == head
        for param in head_module.parameters():
            param.requires_grad = requires
        if requires:
            for m in head_module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    model.to(args.device)

    cfg = Stage3Config(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        label_smoothing=args.label_smoothing,
    )

    class_weights = _class_weights(train_bundle["label"], num_classes)

    # Map to swap labels when applying horizontal flip for AB head
    hflip_swap = None
    rot90_swap = None
    if head == "AB":
        # HORZ_A<->HORZ_B, VERT_A<->VERT_B
        hflip_swap = {0: 1, 1: 0, 2: 3, 3: 2}
        # HORZ <-> VERT when rotating 90°
        rot90_swap = {0: 2, 2: 0, 1: 3, 3: 1}

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train": [], "val": []}

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        preds_list = []
        targets_list = []

        for batch in train_loader:
            images = batch["image"].to(cfg.device)
            labels = batch["label"].to(cfg.device).clone()
            qp = batch.get("qp")
            qp = qp.to(cfg.device) if qp is not None else None

            if random.random() < 0.5:
                images = torch.flip(images, dims=[2])  # vertical
            if random.random() < 0.3:
                images = torch.rot90(images, k=2, dims=[2, 3])  # 180°

            if hflip_swap is not None and random.random() < 0.5:
                images = torch.flip(images, dims=[3])
                labels_orig = labels.clone()
                for src, dst in hflip_swap.items():
                    mask = labels_orig == src
                    labels[mask] = dst

            if rot90_swap is not None and random.random() < 0.3:
                images = torch.rot90(images, k=1, dims=[2, 3])
                labels_orig = labels.clone()
                for src, dst in rot90_swap.items():
                    mask = labels_orig == src
                    labels[mask] = dst

            optimizer.zero_grad()
            outputs = model(image=images, qp=qp)
            logits = outputs.specialists[head]
            loss = F.cross_entropy(logits, labels, weight=class_weights.to(cfg.device), label_smoothing=cfg.label_smoothing)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_examples += images.size(0)
            preds_list.append(torch.argmax(logits.detach(), dim=1).cpu())
            targets_list.append(labels.detach().cpu())

        preds = torch.cat(preds_list)
        targets = torch.cat(targets_list)
        train_metrics = _metrics(preds, targets, num_classes)
        train_metrics.loss = total_loss / max(total_examples, 1)
        history["train"].append(train_metrics.__dict__)

        val_metrics = _evaluate(model, val_loader, cfg, head, num_classes, class_weights)
        history["val"].append(val_metrics.__dict__)

        print(
            f"[Stage3-{head}] Epoch {epoch+1}/{cfg.epochs} | "
            f"Train loss: {train_metrics.loss:.4f}, macro-F1: {train_metrics.macro_f1:.3f} | "
            f"Val loss: {val_metrics.loss:.4f}, macro-F1: {val_metrics.macro_f1:.3f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / f"stage3_{head}_history_block{args.block_size}.json"
    history_path.write_text(json.dumps(history, indent=2))

    checkpoint_path = args.checkpoint_path or args.output_dir / f"stage3_{head}_model_block{args.block_size}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "head": head,
        "num_classes": num_classes,
    }, checkpoint_path)
    print(f"[Stage3-{head}] Treinamento concluído. Checkpoint -> {checkpoint_path}")


if __name__ == "__main__":
    main()
