"""
Script 007: Optimize Stage 1 Threshold
Grid search to find optimal threshold for Stage 1 binary classification.
Evaluates precision, recall, and F1 across different threshold values.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Add v6 pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "v6_pipeline"))
from data_hub import HierarchicalBlockDatasetV6, BlockRecord, build_hierarchical_dataset_v6
from models import Stage1Model


def evaluate_with_threshold(model, dataloader, device, threshold):
    """Evaluate model with a specific threshold"""
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label_stage1'].to(device)
            
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze()
            
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Apply threshold
    predictions = (all_probs >= threshold).astype(int)
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average='binary', pos_label=1, zero_division=0
    )
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
    
    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize Stage 1 Threshold")
    parser.add_argument("--dataset-dir", type=str,
                       default="pesquisa_v6/v6_dataset/block_16",
                       help="Dataset directory")
    parser.add_argument("--model-path", type=str,
                       default="pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt",
                       help="Path to trained Stage 1 model")
    parser.add_argument("--output-dir", type=str,
                       default="pesquisa_v6/logs/v6_experiments/threshold_optimization",
                       help="Output directory for results")
    parser.add_argument("--threshold-min", type=float, default=0.4,
                       help="Minimum threshold to test")
    parser.add_argument("--threshold-max", type=float, default=0.7,
                       help="Maximum threshold to test")
    parser.add_argument("--threshold-step", type=float, default=0.05,
                       help="Threshold step size")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    args = parser.parse_args()
    
    # Setup paths
    dataset_dir = Path(args.dataset_dir)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"  Stage 1 Threshold Optimization")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_dir}")
    print(f"  Threshold range: [{args.threshold_min}, {args.threshold_max}] step {args.threshold_step}")
    
    # Load validation dataset
    print(f"\n[1/4] Loading validation dataset...")
    val_data = torch.load(dataset_dir / "val.pt")
    print(f"  Validation samples: {len(val_data['samples'])}")
    
    from augmentation import Stage1Augmentation
    val_aug = Stage1Augmentation(train=False)
    
    val_record = BlockRecord(
        samples=val_data['samples'].numpy().transpose(0, 2, 3, 1),
        labels=val_data['labels_stage1'].numpy(),
        qps=val_data['qps'].numpy().reshape(-1, 1)
    )
    val_dataset = build_hierarchical_dataset_v6(val_record, augmentation=val_aug, stage='stage1')
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Load model
    print(f"\n[2/4] Loading model...")
    model = Stage1Model(pretrained=False).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    
    # Grid search
    print(f"\n[3/4] Running grid search...")
    thresholds = np.arange(args.threshold_min, args.threshold_max + args.threshold_step, args.threshold_step)
    results = []
    
    pbar = tqdm(thresholds, desc="Testing thresholds")
    for threshold in pbar:
        metrics = evaluate_with_threshold(model, val_loader, device, threshold)
        results.append(metrics)
        pbar.set_postfix({
            'F1': f"{metrics['f1']:.3f}",
            'Prec': f"{metrics['precision']:.3f}",
            'Rec': f"{metrics['recall']:.3f}"
        })
    
    # Find best thresholds
    print(f"\n[4/4] Analyzing results...")
    df = pd.DataFrame(results)
    
    # Best by different criteria
    best_f1_idx = df['f1'].idxmax()
    best_precision_idx = df['precision'].idxmax()
    best_recall_idx = df['recall'].idxmax()
    best_accuracy_idx = df['accuracy'].idxmax()
    
    best_f1 = df.loc[best_f1_idx]
    best_precision = df.loc[best_precision_idx]
    best_recall = df.loc[best_recall_idx]
    best_accuracy = df.loc[best_accuracy_idx]
    
    # Print results
    print(f"\n{'='*70}")
    print(f"  Optimization Results")
    print(f"{'='*70}")
    
    print(f"\n  Best F1 Score:")
    print(f"    Threshold: {best_f1['threshold']:.2f}")
    print(f"    F1: {best_f1['f1']:.2%}")
    print(f"    Precision: {best_f1['precision']:.2%}")
    print(f"    Recall: {best_f1['recall']:.2%}")
    print(f"    Accuracy: {best_f1['accuracy']:.2%}")
    
    print(f"\n  Best Precision:")
    print(f"    Threshold: {best_precision['threshold']:.2f}")
    print(f"    Precision: {best_precision['precision']:.2%}")
    print(f"    F1: {best_precision['f1']:.2%}")
    print(f"    Recall: {best_precision['recall']:.2%}")
    
    print(f"\n  Best Recall:")
    print(f"    Threshold: {best_recall['threshold']:.2f}")
    print(f"    Recall: {best_recall['recall']:.2%}")
    print(f"    F1: {best_recall['f1']:.2%}")
    print(f"    Precision: {best_recall['precision']:.2%}")
    
    print(f"\n  Best Accuracy:")
    print(f"    Threshold: {best_accuracy['threshold']:.2f}")
    print(f"    Accuracy: {best_accuracy['accuracy']:.2%}")
    print(f"    F1: {best_accuracy['f1']:.2%}")
    
    # Default threshold (0.5) performance
    default_result = df[df['threshold'] == 0.5].iloc[0] if 0.5 in df['threshold'].values else None
    if default_result is not None:
        print(f"\n  Default Threshold (0.5):")
        print(f"    F1: {default_result['f1']:.2%}")
        print(f"    Precision: {default_result['precision']:.2%}")
        print(f"    Recall: {default_result['recall']:.2%}")
    
    # Save results
    print(f"\n{'='*70}")
    print(f"  Saving Results")
    print(f"{'='*70}")
    
    # Save CSV
    csv_path = output_dir / "threshold_optimization_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ CSV saved: {csv_path}")
    
    # Save JSON summary
    summary = {
        'best_f1': best_f1.to_dict(),
        'best_precision': best_precision.to_dict(),
        'best_recall': best_recall.to_dict(),
        'best_accuracy': best_accuracy.to_dict(),
        'default_threshold': default_result.to_dict() if default_result is not None else None,
        'all_results': results,
        'config': vars(args)
    }
    
    json_path = output_dir / "threshold_optimization_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ Summary saved: {json_path}")
    
    # Recommendation
    print(f"\n{'='*70}")
    print(f"  Recommendation")
    print(f"{'='*70}")
    print(f"  For balanced F1: Use threshold {best_f1['threshold']:.2f}")
    print(f"  For high precision: Use threshold {best_precision['threshold']:.2f}")
    print(f"  For high recall: Use threshold {best_recall['threshold']:.2f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
