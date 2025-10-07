"""Data utilities for the hierarchical AV1 partition pipeline (v5).

This module reorganises the monolithic logic from `training_v5.ipynb`
into callable helpers that (1) index raw experiment files, (2) load
block samples/labels/QPs into memory, and (3) expose torch Datasets
with hierarchical labels for the new multi-stage CNN flow.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants describing AV1 partition labels and hierarchical groupings
# ---------------------------------------------------------------------------
PARTITION_ID_TO_NAME: Dict[int, str] = {
    0: "PARTITION_NONE",
    1: "PARTITION_HORZ",
    2: "PARTITION_VERT",
    3: "PARTITION_SPLIT",
    4: "PARTITION_HORZ_A",
    5: "PARTITION_HORZ_B",
    6: "PARTITION_VERT_A",
    7: "PARTITION_VERT_B",
    8: "PARTITION_HORZ_4",
    9: "PARTITION_VERT_4",
}

STAGE2_GROUPS: Dict[str, Tuple[str, ...]] = {
    "NONE": ("PARTITION_NONE",),
    "SPLIT": ("PARTITION_SPLIT",),
    "RECT": ("PARTITION_HORZ", "PARTITION_VERT"),
    "AB": (
        "PARTITION_HORZ_A",
        "PARTITION_HORZ_B",
        "PARTITION_VERT_A",
        "PARTITION_VERT_B",
    ),
    "1TO4": ("PARTITION_HORZ_4", "PARTITION_VERT_4"),
}

# Stage-3 specialist heads reference this mapping
STAGE3_GROUPS: Dict[str, Tuple[str, ...]] = {
    "RECT": ("PARTITION_HORZ", "PARTITION_VERT"),
    "AB": (
        "PARTITION_HORZ_A",
        "PARTITION_HORZ_B",
        "PARTITION_VERT_A",
        "PARTITION_VERT_B",
    ),
    "1TO4": ("PARTITION_HORZ_4", "PARTITION_VERT_4"),
}

BLOCK_SIZES = ("8", "16", "32", "64")

# ---------------------------------------------------------------------------
# Raw file discovery
# ---------------------------------------------------------------------------

def index_sequences(base_path: Path) -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
    """Enumerate sample/label/QP triplets per sequence and block size."""

    base_path = Path(base_path).expanduser().resolve()
    dirs = {
        "samples": base_path / "intra_raw_blocks",
        "labels": base_path / "labels",
        "qps": base_path / "qps",
    }
    for name, folder in dirs.items():
        if not folder.is_dir():
            raise FileNotFoundError(f"Required directory missing: {folder} ({name})")

    inventory: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
    sample_files = sorted(p for p in dirs["samples"].iterdir() if p.suffix == ".txt")

    sequence_names = {
        path.name.replace(".txt", "").split("_sample_")[0]
        for path in sample_files
        if "_sample_" in path.name
    }

    dir_map = {
        "sample": dirs["samples"],
        "label": dirs["labels"],
        "qps": dirs["qps"],
    }

    for seq_name in sorted(sequence_names):
        inventory[seq_name] = {}
        for block in BLOCK_SIZES:
            entry = {
                "sample": f"{seq_name}_sample_{block}.txt",
                "label": f"{seq_name}_labels_{block}_intra.txt",
                "qps": f"{seq_name}_qps_{block}_intra.txt",
            }
            resolved = {}
            for key, file in entry.items():
                folder = dir_map[key]
                resolved[key] = file if (folder / file).exists() else None
            inventory[seq_name][block] = resolved
    return inventory


@dataclass
class BlockRecord:
    """Holds raw numpy arrays for a single block size."""

    samples: np.ndarray  # (N, block_size, block_size, C)
    labels: np.ndarray   # (N,)
    qps: np.ndarray      # (N, 1)

    @property
    def block_size(self) -> int:
        return self.samples.shape[1]

    def to_torch(self) -> "TorchBlockRecord":
        # Normalize 10-bit data (0-1023) to [0, 1] range
        torchvision_order = np.transpose(self.samples, (0, 3, 1, 2)).astype(np.float32) / 1023.0
        return TorchBlockRecord(
            samples=torch.from_numpy(torchvision_order),
            labels=torch.from_numpy(self.labels.astype(np.int64)),
            qps=torch.from_numpy(self.qps.squeeze(-1).astype(np.float32)),
        )


@dataclass
class TorchBlockRecord:
    samples: torch.Tensor  # (N, C, H, W)
    labels: torch.Tensor   # int64
    qps: torch.Tensor      # float32


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_block_records(base_path: Path, block_size: str) -> BlockRecord:
    """Load every sample/label/qp tuple for a given block size into memory."""

    if block_size not in BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {BLOCK_SIZES}, got {block_size}")

    base_path = Path(base_path)
    index = index_sequences(base_path)

    samples: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    qps: List[np.ndarray] = []

    for seq_name, blocks in index.items():
        entry = blocks.get(block_size)
        if not entry:
            continue

        sample_file = entry.get("sample")
        label_file = entry.get("label")
        qp_file = entry.get("qps")
        if not (sample_file and label_file and qp_file):
            continue

        with open(base_path / "intra_raw_blocks" / sample_file, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint16)
        block = int(block_size)
        sample_array = raw.reshape(-1, block, block, 1)
        samples.append(sample_array)

        label_array = np.fromfile(base_path / "labels" / label_file, dtype=np.uint8, sep=" ")
        labels.append(label_array.reshape(-1))

        qp_array = np.fromfile(base_path / "qps" / qp_file, dtype=np.uint8, sep=" ")
        qps.append(qp_array.reshape(-1, 1))

    if not samples:
        raise RuntimeError(f"No samples found for block size {block_size}")

    stacked_samples = np.concatenate(samples, axis=0)
    stacked_labels = np.concatenate(labels, axis=0)
    stacked_qps = np.concatenate(qps, axis=0)

    return BlockRecord(
        samples=stacked_samples,
        labels=stacked_labels,
        qps=stacked_qps,
    )


def train_test_split(record: BlockRecord, test_ratio: float = 0.2, seed: int = 42) -> Tuple[BlockRecord, BlockRecord]:
    """Shuffle and split in-memory arrays into train/test partitions."""

    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    total = record.samples.shape[0]
    indices = rng.permutation(total)
    split_point = int(total * (1 - test_ratio))

    def subset(idxs: np.ndarray) -> BlockRecord:
        return BlockRecord(
            samples=record.samples[idxs],
            labels=record.labels[idxs],
            qps=record.qps[idxs],
        )

    train_idx, test_idx = indices[:split_point], indices[split_point:]
    return subset(train_idx), subset(test_idx)


# ---------------------------------------------------------------------------
# Hierarchical label mapping utilities
# ---------------------------------------------------------------------------

PARTITION_NAME_TO_ID = {name: idx for idx, name in PARTITION_ID_TO_NAME.items()}

STAGE2_NAME_TO_ID = {name: i for i, name in enumerate(STAGE2_GROUPS.keys())}

STAGE3_NAME_TO_ID = {
    head: {label: i for i, label in enumerate(group)} for head, group in STAGE3_GROUPS.items()
}


def map_to_stage2(label_ids: np.ndarray) -> np.ndarray:
    names = np.vectorize(PARTITION_ID_TO_NAME.get)(label_ids)
    mapped = np.zeros_like(label_ids)
    for group_name, members in STAGE2_GROUPS.items():
        mask = np.isin(names, members)
        mapped[mask] = STAGE2_NAME_TO_ID[group_name]
    return mapped


def map_to_stage1(label_ids: np.ndarray) -> np.ndarray:
    names = np.vectorize(PARTITION_ID_TO_NAME.get)(label_ids)
    return (names != "PARTITION_NONE").astype(np.uint8)


def map_to_stage3(label_ids: np.ndarray) -> Dict[str, np.ndarray]:
    names = np.vectorize(PARTITION_ID_TO_NAME.get)(label_ids)
    result: Dict[str, np.ndarray] = {}
    for head, members in STAGE3_GROUPS.items():
        head_labels = np.full(label_ids.shape, fill_value=-1, dtype=np.int64)
        for idx, member in enumerate(members):
            head_labels[names == member] = idx
        result[head] = head_labels
    return result


# ---------------------------------------------------------------------------
# Dataset implementation
# ---------------------------------------------------------------------------


class HierarchicalBlockDataset(Dataset):
    """PyTorch dataset that returns hierarchical labels for a single block size."""

    def __init__(
        self,
        record: TorchBlockRecord,
        stage1_labels: torch.Tensor,
        stage2_labels: torch.Tensor,
        stage3_labels: Dict[str, torch.Tensor],
    ) -> None:
        self.samples = record.samples
        self.labels_stage0 = record.labels
        self.qps = record.qps
        self.labels_stage1 = stage1_labels
        self.labels_stage2 = stage2_labels
        self.labels_stage3 = stage3_labels

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: int):
        item = {
            "image": self.samples[idx],
            "qp": self.qps[idx],
            "label_stage0": self.labels_stage0[idx],
            "label_stage1": self.labels_stage1[idx],
            "label_stage2": self.labels_stage2[idx],
        }
        for head, labels in self.labels_stage3.items():
            item[f"label_stage3_{head}"] = labels[idx]
        return item


def build_hierarchical_dataset(record: BlockRecord) -> HierarchicalBlockDataset:
    torch_record = record.to_torch()
    stage1_np = map_to_stage1(record.labels)
    stage2_np = map_to_stage2(record.labels)
    stage3_np = map_to_stage3(record.labels)

    stage3_tensors = {
        head: torch.from_numpy(values.astype(np.int64)) for head, values in stage3_np.items()
    }
    return HierarchicalBlockDataset(
        record=torch_record,
        stage1_labels=torch.from_numpy(stage1_np.astype(np.int64)),
        stage2_labels=torch.from_numpy(stage2_np.astype(np.int64)),
        stage3_labels=stage3_tensors,
    )


# ---------------------------------------------------------------------------
# Metadata persistence helpers
# ---------------------------------------------------------------------------


def save_metadata(path: Path, info: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)


def compute_class_distribution(labels: Iterable[int]) -> Dict[str, float]:
    labels = list(labels)
    total = len(labels)
    counts: Dict[str, int] = {}
    for label in labels:
        name = PARTITION_ID_TO_NAME.get(int(label), "UNKNOWN")
        counts[name] = counts.get(name, 0) + 1
    return {name: count / total for name, count in counts.items()}


__all__ = [
    "BLOCK_SIZES",
    "BlockRecord",
    "PARTITION_ID_TO_NAME",
    "PARTITION_NAME_TO_ID",
    "HierarchicalBlockDataset",
    "STAGE2_GROUPS",
    "STAGE2_NAME_TO_ID",
    "STAGE3_GROUPS",
    "build_hierarchical_dataset",
    "compute_class_distribution",
    "index_sequences",
    "load_block_records",
    "save_metadata",
    "train_test_split",
]
