#!/usr/bin/env python3
"""Train the Stage-2 macro-class classifier for the AV1 hierarchical pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pesquisa.v5_pipeline import (  # noqa: E402
    STAGE2_GROUPS,
    Stage2Config,
    Stage2Metrics,
    build_hierarchical_model,
    train_stage2,
)


class TensorDictDataset(Dataset):
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self._tensors = tensors
        self._length = next(iter(tensors.values())).size(0)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self._tensors.items()}


def make_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    counts = torch.bincount(labels)
    total = counts.sum().item()
    weights_per_class = torch.where(
        counts > 0,
        total / (len(counts) * counts.float()),
        torch.zeros_like(counts, dtype=torch.float32),
    )
    weights = weights_per_class[labels]
    return WeightedRandomSampler(weights, num_samples=labels.numel(), replacement=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("pesquisa/v5_dataset"))
    parser.add_argument("--block-size", choices=["8", "16", "32", "64"], default="16")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("pesquisa/logs/v5_stage2"))
    parser.add_argument("--use-sampler", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--stage1-checkpoint", type=Path, default=None, help="Optional checkpoint from Stage1 training")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-stage1-head", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    return parser.parse_args()


def build_class_weights(stage2_distribution: Dict[str, float]) -> torch.Tensor:
    probs = torch.tensor(
        [stage2_distribution.get(name, 0.0) for name in STAGE2_GROUPS.keys()],
        dtype=torch.float32,
    )
    weights = torch.where(probs > 0, 1.0 / probs, torch.zeros_like(probs))
    return weights


def main() -> None:
    args = parse_args()
    block_dir = args.dataset_root / f"block_{args.block_size}"
    if not block_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {block_dir}")

    train_bundle = torch.load(block_dir / "train.pt")
    val_bundle = torch.load(block_dir / "val.pt")
    metadata = json.loads(Path(block_dir / "metadata.json").read_text())

    train_dataset = TensorDictDataset(train_bundle)
    val_dataset = TensorDictDataset(val_bundle)

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
    }
    if args.use_sampler:
        train_loader_kwargs["sampler"] = make_sampler(train_bundle["label_stage2"])  # type: ignore[index]
    else:
        train_loader_kwargs["shuffle"] = True

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    stage2_classes = len(STAGE2_GROUPS)
    model = build_hierarchical_model(
        stage2_classes=stage2_classes,
        specialist_classes={},
        use_qp=False,
    )

    if args.stage1_checkpoint is not None and args.stage1_checkpoint.exists():
        payload = torch.load(args.stage1_checkpoint, map_location="cpu")
        state_dict = payload.get("model_state", payload)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("stage2_head.")}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Stage2] Missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"[Stage2] Unexpected keys when loading checkpoint: {unexpected}")

    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    if args.freeze_stage1_head:
        for param in model.stage1_head.parameters():
            param.requires_grad = False

    class_weights = build_class_weights(metadata["train"]["stage2_distribution"])

    cfg = Stage2Config(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        device=args.device,
        num_classes=stage2_classes,
    )

    history = train_stage2(
        model,
        loaders={"train": train_loader, "val": val_loader},
        cfg=cfg,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / f"stage2_history_block{args.block_size}.pt"
    torch.save(history, history_path)

    metrics_path = args.output_dir / f"stage2_metrics_block{args.block_size}.json"
    serialised = {
        split: [metric.__dict__ for metric in metrics] for split, metrics in history.items()
    }
    metrics_path.write_text(json.dumps(serialised, indent=2))

    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else args.output_dir / f"stage2_model_block{args.block_size}.pt"
    )
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "metadata": metadata,
    }, checkpoint_path)
    print(
        "Stage2 training complete. Metrics -> "
        f"{metrics_path}, checkpoint -> {checkpoint_path}")


if __name__ == "__main__":
    main()
