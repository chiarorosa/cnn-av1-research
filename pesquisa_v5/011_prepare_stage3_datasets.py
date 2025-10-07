#!/usr/bin/env python3
"""Prepare specialist datasets (Stage3) for RECT, AB, 1TO4 macro classes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pesquisa.v5_pipeline.data_hub import (  # noqa: E402
    STAGE2_GROUPS,
    STAGE2_NAME_TO_ID,
    STAGE3_GROUPS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("pesquisa/v5_dataset_partitioned"),
        help="Diretório com os tensores Stage2 (partitioned)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pesquisa/v5_dataset_stage3"),
        help="Destino para salvar os datasets dos especialistas",
    )
    parser.add_argument(
        "--block-size",
        choices=["8", "16", "32", "64"],
        nargs="*",
        default=["16"],
    )
    parser.add_argument(
        "--heads",
        choices=["RECT", "AB", "1TO4"],
        nargs="*",
        default=["RECT", "AB", "1TO4"],
        help="Especialistas a preparar",
    )
    return parser.parse_args()


def _class_distribution(labels: torch.Tensor, num_classes: int) -> Dict[str, float]:
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum().clamp(min=1.0)
    return {str(i): (counts[i] / total).item() for i in range(num_classes)}


def _filter_bundle(bundle: Dict[str, torch.Tensor], head: str) -> Dict[str, torch.Tensor]:
    stage2_target = STAGE2_NAME_TO_ID[head]
    stage2_mask = bundle["label_stage2"] == stage2_target
    stage3_key = f"label_stage3_{head}"
    if stage3_key not in bundle:
        raise KeyError(f"Bundle não possui rótulos Stage3 para {head}")
    stage3_labels = bundle[stage3_key]
    valid_mask = stage3_labels >= 0
    mask = stage2_mask & valid_mask

    filtered = {
        "image": bundle["image"][mask],
        "qp": bundle["qp"][mask],
        "label": stage3_labels[mask],
    }
    return filtered


def main() -> None:
    args = parse_args()
    heads = args.heads
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for block in args.block_size:
        block_name = f"block_{block}"
        src_block_dir = args.source_root / block_name
        if not src_block_dir.exists():
            raise FileNotFoundError(f"Diretório de origem não encontrado: {src_block_dir}")

        train_bundle = torch.load(src_block_dir / "train.pt")
        val_bundle = torch.load(src_block_dir / "val.pt")

        for head in heads:
            head_dir = output_root / head / block_name
            head_dir.mkdir(parents=True, exist_ok=True)

            filtered_train = _filter_bundle(train_bundle, head)
            filtered_val = _filter_bundle(val_bundle, head)

            torch.save(filtered_train, head_dir / "train.pt")
            torch.save(filtered_val, head_dir / "val.pt")

            num_classes = len(STAGE3_GROUPS[head])
            meta = {
                "head": head,
                "block_size": block,
                "num_classes": num_classes,
                "train_samples": int(filtered_train["label"].shape[0]),
                "val_samples": int(filtered_val["label"].shape[0]),
                "train_distribution": _class_distribution(filtered_train["label"], num_classes),
                "val_distribution": _class_distribution(filtered_val["label"], num_classes),
            }
            (head_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
            print(f"[Stage3] {head} {block_name}: train={meta['train_samples']} val={meta['val_samples']}")


if __name__ == "__main__":
    main()
