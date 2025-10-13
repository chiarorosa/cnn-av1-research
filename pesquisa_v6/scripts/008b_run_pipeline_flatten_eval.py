#!/usr/bin/env python3
"""
008b_run_pipeline_flatten_eval.py

Evaluate the Flatten Pipeline (Stage 1 → Stage 2 Flat 7-way) on validation set.

Pipeline:
    Stage 1: NONE vs PARTITION (binary, threshold 0.45)
        └─ If NONE → output PARTITION_NONE
        └─ If PARTITION → Stage 2 Flat
    
    Stage 2 Flat: 7-way direct classification
        └─ HORZ, VERT, SPLIT, HORZ_A, HORZ_B, VERT_A, VERT_B

Architecture Motivation:
    - Eliminates Stage 3 cascade error from V6 hierarchical pipeline
    - Baseline: 47.66% accuracy (v6 hierarchical with -95% Stage 3 degradation)
    - Target: ≥53% accuracy (success), 50-53% (acceptable), <50% (pivot)

Evaluation:
    - Full validation set (90,793 samples, 10 classes)
    - Stage 1 binary threshold: 0.45 (default from training)
    - Stage 2 Flat 7-class predictions remapped to 10-class space
    - Metrics: accuracy, macro F1, per-class F1, confusion matrix

Usage:
    python3 008b_run_pipeline_flatten_eval.py \\
        --dataset-dir pesquisa_v6/v6_dataset/block_16 \\
        --stage1-model pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt \\
        --stage2-flat-model pesquisa_v6/logs/v6_experiments/stage2_flat/stage2_flat_model_best.pt \\
        --stage1-threshold 0.45 \\
        --batch-size 256 \\
        --output-dir pesquisa_v6/logs/v6_pipeline_flatten_eval \\
        --device cuda
"""

import sys
import os
from pathlib import Path

# Add v6_pipeline to path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent  # pesquisa_v6/
pipeline_dir = project_root / "v6_pipeline"
sys.path.insert(0, str(pipeline_dir))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_hub import (
    PARTITION_ID_TO_NAME,
    FLATTEN_ID_TO_NAME,
)
from metrics import compute_metrics


class HierarchicalBlockDatasetV6(Dataset):
    """Load V6 hierarchical dataset (10 classes with stage0/stage2/stage3 labels)"""
    
    def __init__(self, data_path: Path):
        data = torch.load(data_path, weights_only=False)  # Explicit for compatibility
        
        self.samples = data['samples']  # (N, 1, 16, 16)
        # labels_stage0 contains original 10-class labels (0-9)
        self.original_labels = data['labels_stage0']  # 10-way ground truth
        self.qps = data.get('qps', torch.zeros(len(self.samples)))
        
        # Derive binary labels: 0=NONE, 1=any PARTITION (1-9)
        self.binary_labels = (self.original_labels > 0).long()
        
        print(f"    Samples: {len(self.samples)}")
        print(f"    Original labels (10-way): {self.original_labels.unique().tolist()}")
        print(f"    Binary labels: {self.binary_labels.unique().tolist()}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {
            'sample': self.samples[idx],
            'binary_label': self.binary_labels[idx],
            'original_label': self.original_labels[idx],
            'qp': self.qps[idx]
        }


def load_stage1_model(model_path: Path, device: str) -> nn.Module:
    """Load Stage 1 binary model (NONE vs PARTITION)"""
    from models import Stage1Model
    
    model = Stage1Model(pretrained=False)
    
    # Load checkpoint (weights_only=False for compatibility with numpy)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def load_stage2_flat_model(model_path: Path, device: str) -> nn.Module:
    """Load Stage 2 Flat 7-way model"""
    from models import ImprovedBackbone
    
    # Rebuild Stage2FlatModel architecture
    class Stage2FlatModel(nn.Module):
        def __init__(self, pretrained=True):
            super().__init__()
            self.backbone = ImprovedBackbone(pretrained=pretrained)
            
            # 7-class head (HORZ through VERT_B, no NONE)
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 7)
            )
        
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)
    
    model = Stage2FlatModel(pretrained=False)
    
    # Load checkpoint (weights_only=False for compatibility with numpy)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def remap_flatten_to_original(flatten_label: int) -> int:
    """
    Remap 7-class flatten label to 10-class original label.
    
    Flatten space (7 classes, 0-6):
        0: PARTITION_HORZ
        1: PARTITION_VERT
        2: PARTITION_SPLIT
        3: PARTITION_HORZ_A
        4: PARTITION_HORZ_B
        5: PARTITION_VERT_A
        6: PARTITION_VERT_B
    
    Original space (10 classes, 0-9):
        0: PARTITION_NONE (handled separately in pipeline)
        1: PARTITION_HORZ
        2: PARTITION_VERT
        3: PARTITION_SPLIT
        4: PARTITION_HORZ_A
        5: PARTITION_HORZ_B
        6: PARTITION_VERT_A
        7: PARTITION_VERT_B
        8: PARTITION_HORZ_4 (doesn't exist in dataset)
        9: PARTITION_VERT_4 (doesn't exist in dataset)
    """
    # Flatten labels are shifted by 1 (because NONE is removed)
    return flatten_label + 1


def run_pipeline_inference(
    stage1_model: nn.Module,
    stage2_flat_model: nn.Module,
    dataloader: DataLoader,
    stage1_threshold: float,
    device: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run full pipeline inference.
    
    Returns:
        predictions: (N,) array of predicted labels (10-class space)
        ground_truth: (N,) array of ground truth labels (10-class space)
    """
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Pipeline inference"):
            samples = batch['sample'].to(device)
            original_labels = batch['original_label'].cpu().numpy()
            
            # Stage 1: NONE vs PARTITION
            stage1_logits = stage1_model(samples)
            stage1_probs = torch.sigmoid(stage1_logits).squeeze()
            
            # Apply threshold
            partition_mask = stage1_probs >= stage1_threshold
            
            # Initialize predictions as NONE (label 0)
            batch_preds = torch.zeros(len(samples), dtype=torch.long)
            
            # Stage 2 Flat: Only for PARTITION samples
            if partition_mask.any():
                partition_samples = samples[partition_mask]
                stage2_logits = stage2_flat_model(partition_samples)
                stage2_preds = stage2_logits.argmax(dim=1)
                
                # Remap to original 10-class space
                stage2_preds_remapped = torch.tensor(
                    [remap_flatten_to_original(p.item()) for p in stage2_preds],
                    dtype=torch.long
                )
                
                batch_preds[partition_mask] = stage2_preds_remapped
            
            all_preds.append(batch_preds.cpu().numpy())
            all_labels.append(original_labels)
    
    predictions = np.concatenate(all_preds)
    ground_truth = np.concatenate(all_labels)
    
    return predictions, ground_truth


def compute_pipeline_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    output_dir: Path
):
    """
    Compute and save pipeline metrics.
    
    Metrics:
        - Overall accuracy
        - Macro F1 (average across all classes)
        - Weighted F1 (weighted by class support)
        - Per-class F1, precision, recall
        - Confusion matrix
    """
    # Compute metrics using v6_pipeline.metrics
    metrics = compute_metrics(ground_truth, predictions)
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(ground_truth, predictions, labels=list(range(10)))
    
    # Print results
    print("\n" + "=" * 70)
    print("  Pipeline Flatten Evaluation Results")
    print("=" * 70)
    print(f"\n  Overall Metrics:")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    Macro F1: {metrics['macro_f1']:.4f}")
    print(f"    Weighted F1: {metrics['weighted_f1']:.4f}")
    
    print(f"\n  Per-class F1:")
    for class_id in range(10):
        class_name = PARTITION_ID_TO_NAME.get(class_id, f"Class_{class_id}")
        class_metrics = metrics['per_class'].get(f'class_{class_id}', {})
        f1 = class_metrics.get('f1', 0.0)
        support = class_metrics.get('support', 0)
        print(f"    {class_name:20s}: F1={f1:.4f} (n={support})")
    
    print("=" * 70)
    
    # Build results dict
    results = {
        'overall': {
            'accuracy': float(metrics['accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'weighted_f1': float(metrics['weighted_f1'])
        },
        'per_class': {}
    }
    
    for class_id in range(10):
        class_name = PARTITION_ID_TO_NAME.get(class_id, f"Class_{class_id}")
        class_metrics = metrics['per_class'].get(f'class_{class_id}', {})
        results['per_class'][class_name] = {
            'f1': float(class_metrics.get('f1', 0.0)),
            'precision': float(class_metrics.get('precision', 0.0)),
            'recall': float(class_metrics.get('recall', 0.0)),
            'support': int(class_metrics.get('support', 0))
        }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / 'pipeline_flatten_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    
    conf_matrix_path = output_dir / 'confusion_matrix.npy'
    np.save(conf_matrix_path, conf_matrix)
    print(f"  Confusion matrix saved to: {conf_matrix_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Flatten Pipeline")
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help="Path to v6_dataset/block_16 directory")
    parser.add_argument('--stage1-model', type=str, required=True,
                        help="Path to Stage 1 binary model checkpoint")
    parser.add_argument('--stage2-flat-model', type=str, required=True,
                        help="Path to Stage 2 Flat model checkpoint")
    parser.add_argument('--stage1-threshold', type=float, default=0.45,
                        help="Stage 1 threshold for NONE vs PARTITION (default: 0.45)")
    parser.add_argument('--batch-size', type=int, default=256,
                        help="Batch size for inference (default: 256)")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Directory to save results")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: cuda or cpu (default: cuda)")
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 70)
    print("  Flatten Pipeline Evaluation")
    print("=" * 70)
    print(f"\n  Configuration:")
    print(f"    Dataset: {args.dataset_dir}")
    print(f"    Stage 1 model: {args.stage1_model}")
    print(f"    Stage 2 Flat model: {args.stage2_flat_model}")
    print(f"    Stage 1 threshold: {args.stage1_threshold}")
    print(f"    Batch size: {args.batch_size}")
    print(f"    Device: {args.device}")
    print(f"    Output: {args.output_dir}\n")
    
    # Convert paths
    dataset_dir = Path(args.dataset_dir)
    dataset_path = dataset_dir / 'val.pt'
    output_dir = Path(args.output_dir)
    
    # [1/5] Load dataset
    print("[1/5] Loading validation dataset...")
    dataset = HierarchicalBlockDatasetV6(dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # [2/5] Load models
    print("\n[2/5] Loading models...")
    stage1_model = load_stage1_model(Path(args.stage1_model), args.device)
    print("    Stage 1 model loaded")
    stage2_flat_model = load_stage2_flat_model(Path(args.stage2_flat_model), args.device)
    print("    Stage 2 Flat model loaded")
    
    # [3/5] Run pipeline inference
    print("\n[3/5] Running pipeline inference...")
    predictions, ground_truth = run_pipeline_inference(
        stage1_model,
        stage2_flat_model,
        dataloader,
        args.stage1_threshold,
        args.device
    )
    
    # [4/5] Compute metrics
    print("\n[4/5] Computing metrics...")
    results = compute_pipeline_metrics(predictions, ground_truth, output_dir)
    
    # [5/5] Print verdict
    print("\n[5/5] Verdict:")
    accuracy = results['overall']['accuracy']
    if accuracy >= 0.53:
        verdict = "✅ SUCCESS: Flatten pipeline beats hierarchical baseline (47.66%)"
        recommendation = "Ready to merge feat/stage2-flatten-9classes → main"
    elif accuracy >= 0.50:
        verdict = "⚠️  ACCEPTABLE: Better than baseline but not optimal"
        recommendation = "Consider threshold optimization or accept as partial success"
    else:
        verdict = "❌ PIVOT NEEDED: Flatten does not beat hierarchical baseline"
        recommendation = "Try Multi-Task or Stage 3 Robust architecture"
    
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    {verdict}")
    print(f"    Recommendation: {recommendation}")
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
