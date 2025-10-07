#!/usr/bin/env python3
"""Train the Stage-1 binary classifier for the AV1 hierarchical pipeline."""
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

from pesquisa.v5_pipeline import (
    STAGE2_GROUPS,
    STAGE3_GROUPS,
    Stage1Config,
    Stage1Metrics,
    build_hierarchical_model,
    train_stage1,
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
    positives = (labels == 1).sum().item()
    negatives = (labels == 0).sum().item()
    total = positives + negatives
    if positives == 0 or negatives == 0:
        weights = torch.ones(total)
    else:
        pos_weight = total / (2 * positives)
        neg_weight = total / (2 * negatives)
        weights = torch.where(labels == 1, pos_weight, neg_weight)
    return WeightedRandomSampler(weights, num_samples=total, replacement=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("pesquisa/v5_dataset"))
    parser.add_argument("--block-size", choices=["8", "16", "32", "64"], default="16")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("pesquisa/logs/v5_stage1"))
    parser.add_argument("--use-sampler", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    return parser.parse_args()


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
        train_loader_kwargs["sampler"] = make_sampler(train_bundle["label_stage1"])
    else:
        train_loader_kwargs["shuffle"] = True

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    stage2_classes = len(STAGE2_GROUPS)
    specialist_classes = {head: len(labels) for head, labels in STAGE3_GROUPS.items()}
    model = build_hierarchical_model(
        stage2_classes=stage2_classes,
        specialist_classes=specialist_classes,
        use_qp=False,
    )

    labels_stage1 = train_bundle["label_stage1"]
    positives = (labels_stage1 == 1).sum().item()
    negatives = (labels_stage1 == 0).sum().item()
    pos_weight = 1.0 if positives == 0 else max(negatives / positives, 1.0)

    cfg = Stage1Config(
        epochs=args.epochs,
        lr=1e-3,
        weight_decay=1e-4,
        pos_weight=pos_weight,
        focal_gamma=args.focal_gamma,
        device=args.device,
    )
    if "stage1_pos_weight" in metadata:
        cfg.pos_weight = float(metadata["stage1_pos_weight"].get("train", cfg.pos_weight))

    history = train_stage1(
        model,
        loaders={"train": train_loader, "val": val_loader},
        cfg=cfg,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / f"stage1_history_block{args.block_size}.pt"
    torch.save(history, history_path)

    metrics_path = args.output_dir / f"stage1_metrics_block{args.block_size}.json"
    serialised = {
        split: [metric.__dict__ for metric in metrics] for split, metrics in history.items()
    }
    metrics_path.write_text(json.dumps(serialised, indent=2))

    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else args.output_dir / f"stage1_model_block{args.block_size}.pt"
    )
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "metadata": metadata,
    }, checkpoint_path)
    print(f"Stage1 training complete. Metrics -> {metrics_path}, checkpoint -> {checkpoint_path}")


if __name__ == "__main__":
    main()
