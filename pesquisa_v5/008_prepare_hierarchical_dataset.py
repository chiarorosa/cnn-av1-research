#!/usr/bin/env python3
"""Prepare hierarchical datasets for the AV1 CNN pipeline (v5).

This script loads raw block samples, derives the hierarchical labels and
persists ready-to-use torch tensors plus metadata for Stage1/Stage2/Stage3
training loops.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pesquisa.v5_pipeline.data_hub import (
    BLOCK_SIZES,
    STAGE2_GROUPS,
    STAGE3_GROUPS,
    build_hierarchical_dataset,
    compute_class_distribution,
    load_block_records,
    save_metadata,
    train_test_split,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stage1_pos_weight(labels: torch.Tensor) -> float:
    positives = (labels == 1).sum().item()
    negatives = (labels == 0).sum().item()
    if positives == 0:
        return 1.0
    return max(negatives / positives, 1.0)


def _collect_metadata(record, stage1, stage2, stage3) -> Dict[str, object]:
    stage2_names = list(STAGE2_GROUPS.keys())
    meta: Dict[str, object] = {
        "block_size": record.block_size,
        "num_samples": int(record.samples.shape[0]),
        "stage0_distribution": compute_class_distribution(record.labels),
        "stage1_distribution": {
            "partitioned": float((stage1 == 1).sum().item() / stage1.numel()),
            "none": float((stage1 == 0).sum().item() / stage1.numel()),
        },
        "stage2_distribution": {
            name: float((stage2 == idx).sum().item() / stage2.numel())
            for idx, name in enumerate(stage2_names)
        },
    }
    stage3_meta = {}
    for head, labels in stage3.items():
        valid = labels >= 0
        if valid.any():
            names = STAGE3_GROUPS[head]
            counts = torch.bincount(labels[valid], minlength=len(names))
            total = valid.sum().item()
            stage3_meta[head] = {
                name: float(counts[idx].item() / total) for idx, name in enumerate(names)
            }
        else:
            stage3_meta[head] = {}
    meta["stage3_distribution"] = stage3_meta
    return meta


def _save_tensor_bundle(path: Path, dataset) -> None:
    payload = {
        "image": dataset.samples,
        "qp": dataset.qps,
        "label_stage0": dataset.labels_stage0,
        "label_stage1": dataset.labels_stage1,
        "label_stage2": dataset.labels_stage2,
    }
    for head, tensor in dataset.labels_stage3.items():
        payload[f"label_stage3_{head}"] = tensor
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/home/chiarorosa/experimentos/ai_ugc"),
        help="Directory containing raw intra samples/labels/qps",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pesquisa/v5_dataset"),
        help="Destination directory for prepared tensors",
    )
    parser.add_argument(
        "--block-size",
        choices=BLOCK_SIZES,
        nargs="*",
        default=None,
        help="Subset of block sizes to process",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--partitioned-only",
        action="store_true",
        help="Mantém apenas blocos cuja label_stage1 seja 1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    block_sizes = args.block_size or BLOCK_SIZES
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for block in block_sizes:
        record = load_block_records(args.base_dir, block)
        train_record, val_record = train_test_split(record, test_ratio=args.val_ratio, seed=args.seed)

        train_dataset = build_hierarchical_dataset(train_record)
        val_dataset = build_hierarchical_dataset(val_record)

        if args.partitioned_only:
            def _filter_partitioned(dataset):
                mask = dataset.labels_stage1 == 1
                dataset.samples = dataset.samples[mask]
                dataset.labels_stage0 = dataset.labels_stage0[mask]
                dataset.qps = dataset.qps[mask]
                dataset.labels_stage1 = dataset.labels_stage1[mask]
                dataset.labels_stage2 = dataset.labels_stage2[mask]
                for head in list(dataset.labels_stage3.keys()):
                    dataset.labels_stage3[head] = dataset.labels_stage3[head][mask]
                return dataset

            train_dataset = _filter_partitioned(train_dataset)
            val_dataset = _filter_partitioned(val_dataset)

        block_dir = output_dir / f"block_{block}"
        block_dir.mkdir(parents=True, exist_ok=True)
        _save_tensor_bundle(block_dir / "train.pt", train_dataset)
        _save_tensor_bundle(block_dir / "val.pt", val_dataset)

        def _metadata_from_dataset(record, dataset):
            data_meta = _collect_metadata(
                record,
                dataset.labels_stage1,
                dataset.labels_stage2,
                dataset.labels_stage3,
            )
            if args.partitioned_only:
                stage0 = compute_class_distribution(dataset.labels_stage0.tolist())
                stage0.pop("PARTITION_NONE", None)
                total_stage0 = sum(stage0.values()) or 1.0
                data_meta["stage0_distribution"] = {
                    name: value / total_stage0 for name, value in stage0.items()
                }
                data_meta["stage1_distribution"] = {"none": 0.0, "partitioned": 1.0}
                stage2 = data_meta["stage2_distribution"]
                total = 1.0 - stage2.get("NONE", 0.0)
                data_meta["stage2_distribution"] = {
                    name: value / total for name, value in stage2.items() if name != "NONE"
                }
            return data_meta

        meta = {
            "train": _metadata_from_dataset(train_record, train_dataset),
            "val": _metadata_from_dataset(val_record, val_dataset),
        }
        meta["stage1_pos_weight"] = {
            "train": _stage1_pos_weight(train_dataset.labels_stage1),
            "val": _stage1_pos_weight(val_dataset.labels_stage1),
        }
        save_metadata(block_dir / "metadata.json", meta)
        summary[block] = meta["train"]["stage0_distribution"]

    save_metadata(output_dir / "summary.json", {"stage0_distribution": summary})


if __name__ == "__main__":
    main()
