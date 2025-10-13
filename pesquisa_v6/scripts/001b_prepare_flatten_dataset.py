#!/usr/bin/env python3
"""
001b - Prepare Flatten Dataset (9 classes directly)

This script creates a simplified dataset for the Flatten architecture:
- Input: V6 dataset (10 classes with hierarchical labels)
- Output: Flatten dataset (9 classes: PARTITION only, NONE removed)

Label mapping (original → flatten):
  PARTITION_NONE     (0) → REMOVED (handled by Stage 1)
  PARTITION_HORZ     (1) → 0
  PARTITION_VERT     (2) → 1
  PARTITION_SPLIT    (3) → 2
  PARTITION_HORZ_A   (4) → 3
  PARTITION_HORZ_B   (5) → 4
  PARTITION_VERT_A   (6) → 5
  PARTITION_VERT_B   (7) → 6
  PARTITION_HORZ_4   (8) → 7
  PARTITION_VERT_4   (9) → 8

Architecture context:
- Stage 1 (Binary): NONE vs PARTITION → maintains 72.79% accuracy
- Stage 2 (Flatten): 9-way classifier directly predicts partition type
- No Stage 3 needed → eliminates cascade error (-95% degradation)

Expected dataset:
- Train: ~38,264 PARTITION samples (filtered from 90,828 total)
- Val: ~38,256 PARTITION samples (filtered from 90,793 total)
- Class imbalance: 96:1 ratio (HORZ 9,618 vs HORZ_4 100 samples)

Author: Chiaro Rosa (PhD Research - AV1 Partition Prediction)
Date: 2025-01
Version: v6-flatten
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Label mapping constants
# ---------------------------------------------------------------------------

# Original 10 AV1 partition types
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

# Flatten mapping: remove NONE (0), remap 1-9 → 0-8
ORIGINAL_TO_FLATTEN: Dict[int, int] = {
    1: 0,  # HORZ
    2: 1,  # VERT
    3: 2,  # SPLIT
    4: 3,  # HORZ_A
    5: 4,  # HORZ_B
    6: 5,  # VERT_A
    7: 6,  # VERT_B
    8: 7,  # HORZ_4
    9: 8,  # VERT_4
}

FLATTEN_ID_TO_NAME: Dict[int, str] = {
    0: "PARTITION_HORZ",
    1: "PARTITION_VERT",
    2: "PARTITION_SPLIT",
    3: "PARTITION_HORZ_A",
    4: "PARTITION_HORZ_B",
    5: "PARTITION_VERT_A",
    6: "PARTITION_VERT_B",
    7: "PARTITION_HORZ_4",
    8: "PARTITION_VERT_4",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_v6_dataset(data_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load V6 dataset (hierarchical labels).
    
    Returns:
        samples: (N, C, H, W) normalized to [0, 1]
        labels_stage0: (N,) original 10-class labels (0-9)
        qps: (N,) quantization parameters
    """
    print(f"  Loading from: {data_path}")
    data = torch.load(data_path)
    
    samples = data['samples']
    labels_stage0 = data['labels_stage0']  # Original 10 classes
    qps = data['qps']
    
    print(f"    Samples shape: {samples.shape}")
    print(f"    Labels shape: {labels_stage0.shape}")
    print(f"    QPs shape: {qps.shape}")
    
    return samples, labels_stage0, qps


def filter_and_remap(
    samples: torch.Tensor,
    labels: torch.Tensor,
    qps: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter PARTITION_NONE (label=0) and remap remaining labels 1-9 → 0-8.
    
    Args:
        samples: (N, C, H, W) all samples
        labels: (N,) original labels 0-9
        qps: (N,) quantization parameters
        
    Returns:
        filtered_samples: (M, C, H, W) where M = samples with label > 0
        remapped_labels: (M,) labels in range 0-8
        filtered_qps: (M,) corresponding QPs
    """
    # 1. Find PARTITION samples (label > 0, i.e., not NONE)
    partition_mask = labels > 0
    num_filtered = partition_mask.sum().item()
    
    print(f"  Filtering NONE samples:")
    print(f"    Total samples: {len(labels)}")
    print(f"    NONE samples (label=0): {(labels == 0).sum().item()}")
    print(f"    PARTITION samples (label>0): {num_filtered}")
    
    # 2. Apply filter
    filtered_samples = samples[partition_mask]
    filtered_labels = labels[partition_mask]
    filtered_qps = qps[partition_mask]
    
    # 3. Remap labels: 1-9 → 0-8
    remapped_labels = torch.zeros_like(filtered_labels)
    for original, flatten in ORIGINAL_TO_FLATTEN.items():
        mask = filtered_labels == original
        remapped_labels[mask] = flatten
    
    # Verify all labels were remapped
    unique_labels = torch.unique(remapped_labels).tolist()
    expected_range = list(range(9))  # 0-8
    
    print(f"  Label remapping:")
    print(f"    Unique remapped labels: {sorted(unique_labels)}")
    print(f"    Expected range: {expected_range}")
    
    if not all(label in expected_range for label in unique_labels):
        raise ValueError(f"Invalid remapped labels: {unique_labels}")
    
    return filtered_samples, remapped_labels, filtered_qps


def compute_class_distribution(labels: torch.Tensor) -> Dict[str, float]:
    """Compute class distribution as percentages."""
    counter = Counter(labels.tolist())
    total = len(labels)
    
    distribution = {}
    for class_id in range(9):  # 0-8
        class_name = FLATTEN_ID_TO_NAME[class_id]
        count = counter.get(class_id, 0)
        percentage = (count / total) * 100 if total > 0 else 0.0
        distribution[class_name] = {
            'count': count,
            'percentage': percentage
        }
    
    return distribution


def save_metadata(output_path: Path, metadata: Dict) -> None:
    """Save metadata as formatted JSON."""
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------

def prepare_flatten_dataset(
    v6_dataset_dir: Path,
    block_size: str = "16",
    output_dir: Path | None = None,
) -> Dict:
    """
    Prepare flatten dataset from V6 hierarchical dataset.
    
    Args:
        v6_dataset_dir: Path to V6 dataset (e.g., pesquisa_v6/v6_dataset/block_16)
        block_size: Block size ("8", "16", "32", "64")
        output_dir: Output directory (default: pesquisa_v6/v6_dataset_flatten/block_{size})
        
    Returns:
        metadata: Dictionary with dataset statistics
    """
    print(f"\n{'='*70}")
    print(f"  Preparing Flatten Dataset (9 classes)")
    print(f"  Block size: {block_size}")
    print(f"{'='*70}\n")
    
    # Set default output dir
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "v6_dataset_flatten" / f"block_{block_size}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    v6_dataset_dir = Path(v6_dataset_dir)
    
    # -----------------------------------------------------------------------
    # 1. Load V6 train dataset
    # -----------------------------------------------------------------------
    print(f"[1/6] Loading V6 train dataset...")
    train_samples, train_labels, train_qps = load_v6_dataset(v6_dataset_dir / "train.pt")
    
    # -----------------------------------------------------------------------
    # 2. Filter and remap train labels
    # -----------------------------------------------------------------------
    print(f"\n[2/6] Filtering and remapping train labels...")
    train_samples_flat, train_labels_flat, train_qps_flat = filter_and_remap(
        train_samples, train_labels, train_qps
    )
    
    # Compute distribution
    train_dist = compute_class_distribution(train_labels_flat)
    print(f"\n  Train distribution ({len(train_labels_flat)} samples):")
    for class_name, stats in train_dist.items():
        print(f"    {class_name:20s}: {stats['count']:6d} ({stats['percentage']:5.2f}%)")
    
    # -----------------------------------------------------------------------
    # 3. Load V6 val dataset
    # -----------------------------------------------------------------------
    print(f"\n[3/6] Loading V6 val dataset...")
    val_samples, val_labels, val_qps = load_v6_dataset(v6_dataset_dir / "val.pt")
    
    # -----------------------------------------------------------------------
    # 4. Filter and remap val labels
    # -----------------------------------------------------------------------
    print(f"\n[4/6] Filtering and remapping val labels...")
    val_samples_flat, val_labels_flat, val_qps_flat = filter_and_remap(
        val_samples, val_labels, val_qps
    )
    
    # Compute distribution
    val_dist = compute_class_distribution(val_labels_flat)
    print(f"\n  Val distribution ({len(val_labels_flat)} samples):")
    for class_name, stats in val_dist.items():
        print(f"    {class_name:20s}: {stats['count']:6d} ({stats['percentage']:5.2f}%)")
    
    # -----------------------------------------------------------------------
    # 5. Save flatten datasets
    # -----------------------------------------------------------------------
    print(f"\n[5/6] Saving flatten datasets...")
    
    train_path = output_dir / "train.pt"
    val_path = output_dir / "val.pt"
    
    print(f"  Saving train to: {train_path}")
    torch.save({
        'samples': train_samples_flat,
        'labels': train_labels_flat,  # Single 9-class labels
        'qps': train_qps_flat,
    }, train_path)
    
    print(f"  Saving val to: {val_path}")
    torch.save({
        'samples': val_samples_flat,
        'labels': val_labels_flat,  # Single 9-class labels
        'qps': val_qps_flat,
    }, val_path)
    
    # -----------------------------------------------------------------------
    # 6. Save metadata
    # -----------------------------------------------------------------------
    print(f"\n[6/6] Saving metadata...")
    
    # Calculate imbalance ratio
    train_counts = [train_dist[name]['count'] for name in FLATTEN_ID_TO_NAME.values()]
    max_count = max(train_counts)
    min_count = min([c for c in train_counts if c > 0])  # Exclude zero counts
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    metadata = {
        'architecture': 'flatten',
        'description': 'Stage 2 predicts 9 classes directly (PARTITION only, NONE filtered)',
        'block_size': block_size,
        'num_classes': 9,
        'class_names': list(FLATTEN_ID_TO_NAME.values()),
        'label_mapping': {
            'original_range': '0-9 (10 classes)',
            'flatten_range': '0-8 (9 classes)',
            'mapping': {PARTITION_ID_TO_NAME[k]: f"{k} → {v}" for k, v in ORIGINAL_TO_FLATTEN.items()},
            'removed': 'PARTITION_NONE (0) handled by Stage 1'
        },
        'train_samples': len(train_labels_flat),
        'val_samples': len(val_labels_flat),
        'train_distribution': train_dist,
        'val_distribution': val_dist,
        'imbalance_stats': {
            'max_class_count': max_count,
            'min_class_count': min_count,
            'imbalance_ratio': f"{imbalance_ratio:.1f}:1",
            'most_frequent': max(train_dist.items(), key=lambda x: x[1]['count'])[0],
            'least_frequent': min(train_dist.items(), key=lambda x: x[1]['count'])[0],
        },
        'source_dataset': str(v6_dataset_dir),
        'output_directory': str(output_dir),
    }
    
    metadata_path = output_dir / "metadata.json"
    save_metadata(metadata_path, metadata)
    print(f"  Saved to: {metadata_path}")
    
    # Print imbalance stats
    print(f"\n  Imbalance statistics:")
    print(f"    Max class count: {max_count}")
    print(f"    Min class count: {min_count}")
    print(f"    Imbalance ratio: {imbalance_ratio:.1f}:1")
    print(f"    Most frequent: {metadata['imbalance_stats']['most_frequent']}")
    print(f"    Least frequent: {metadata['imbalance_stats']['least_frequent']}")
    
    print(f"\n{'='*70}")
    print(f"  ✅ Flatten dataset prepared successfully!")
    print(f"  📁 Output directory: {output_dir}")
    print(f"  📊 Train samples: {len(train_labels_flat)}")
    print(f"  📊 Val samples: {len(val_labels_flat)}")
    print(f"  🎯 Classes: 9 (PARTITION only, NONE removed)")
    print(f"{'='*70}\n")
    
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare Flatten Dataset (9 classes) from V6 Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare from default V6 dataset
  python3 001b_prepare_flatten_dataset.py
  
  # Specify V6 dataset directory
  python3 001b_prepare_flatten_dataset.py \\
      --v6-dataset-dir pesquisa_v6/v6_dataset/block_16
  
  # Custom output directory
  python3 001b_prepare_flatten_dataset.py \\
      --output-dir /path/to/custom/output

Output structure:
  v6_dataset_flatten/
    block_16/
      train.pt      # 9-class PARTITION samples
      val.pt        # 9-class PARTITION samples
      metadata.json # Class distribution, imbalance stats
        """
    )
    
    parser.add_argument(
        "--v6-dataset-dir",
        type=str,
        default=None,
        help="Path to V6 dataset directory (default: pesquisa_v6/v6_dataset/block_16)"
    )
    parser.add_argument(
        "--block-size",
        type=str,
        default="16",
        choices=["8", "16", "32", "64"],
        help="Block size to process (default: 16)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: pesquisa_v6/v6_dataset_flatten/block_<size>)"
    )
    
    args = parser.parse_args()
    
    # Set default V6 dataset directory if not provided
    if args.v6_dataset_dir is None:
        script_dir = Path(__file__).parent.parent
        v6_dataset_dir = script_dir / "v6_dataset" / f"block_{args.block_size}"
    else:
        v6_dataset_dir = Path(args.v6_dataset_dir)
    
    # Verify V6 dataset exists
    if not v6_dataset_dir.exists():
        raise FileNotFoundError(
            f"V6 dataset not found: {v6_dataset_dir}\n"
            f"Please run 001_prepare_v6_dataset.py first."
        )
    
    # Run preparation
    prepare_flatten_dataset(
        v6_dataset_dir=v6_dataset_dir,
        block_size=args.block_size,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
