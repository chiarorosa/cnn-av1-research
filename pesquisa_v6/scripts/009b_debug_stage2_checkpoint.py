#!/usr/bin/env python3
"""
Script 009b: Debug Stage 2 Checkpoint Inconsistency

Objetivo:
    Executar validation loop EXATO do Script 004 e comparar métricas
    com as salvas no history.

Autor: Chiaro Rosa
Data: 14/10/2025
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'v6_pipeline'))

from models import Stage2Model
from data_hub import HierarchicalBlockDatasetV6


def validate_epoch_exact_copy(model, dataloader, criterion, device):
    """
    Cópia EXATA da função validate_epoch() do Script 004.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for blocks, labels, qps in tqdm(dataloader, desc="Validation"):
            blocks = blocks.to(device)
            labels = labels.to(device)
            
            outputs = model(blocks)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * blocks.size(0)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).float().mean().item()
    avg_loss = running_loss / len(dataloader.dataset)
    
    # Per-class metrics
    class_names = ['SPLIT', 'RECT', 'AB']
    per_class_metrics = {}
    
    for cls_idx, cls_name in enumerate(class_names):
        cls_mask = all_labels == cls_idx
        if cls_mask.sum() == 0:
            continue
        
        cls_preds = all_preds[cls_mask]
        cls_labels = all_labels[cls_mask]
        
        tp = ((cls_preds == cls_idx) & (cls_labels == cls_idx)).sum().item()
        fp = ((cls_preds == cls_idx) & (cls_labels != cls_idx)).sum().item()
        fn = ((cls_preds != cls_idx) & (cls_labels == cls_idx)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[cls_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': cls_mask.sum().item()
        }
    
    # Macro F1
    macro_f1 = sum(per_class_metrics[cls]['f1'] for cls in class_names) / len(class_names)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class': per_class_metrics
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset-dir', type=str, default='pesquisa_v6/v6_dataset/block_16')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    
    print("=" * 80)
    print("Script 009b: Debug Stage 2 Checkpoint")
    print("=" * 80)
    
    # Load dataset (EXACT same as Script 004)
    print("\n📂 Loading dataset...")
    dataset = HierarchicalBlockDatasetV6(
        dataset_dir=Path(args.dataset_dir),
        split='val',
        target_stage='stage2',
        augment=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    print(f"  Samples: {len(dataset)}")
    print(f"  Batches: {len(dataloader)}")
    
    # Load model
    print(f"\n🔄 Loading model from: {args.checkpoint}")
    model = Stage2Model(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    print(f"  Checkpoint F1 (saved): {checkpoint['val_macro_f1']:.2%}")
    
    # Load criterion (EXACT same as Script 004)
    from losses import ClassBalancedFocalLoss
    
    val_data = torch.load(Path(args.dataset_dir) / "val.pt", weights_only=False)
    labels_stage2 = val_data['labels_stage2']
    valid_mask = labels_stage2 != -1
    labels_filtered = labels_stage2[valid_mask]
    unique, counts = torch.unique(labels_filtered, return_counts=True)
    samples_per_cls = counts.tolist()
    
    criterion = ClassBalancedFocalLoss(
        samples_per_class=samples_per_cls,
        beta=0.9999,
        gamma=2.0
    ).to(args.device)
    
    # Run validation (EXACT same loop)
    print(f"\n🔄 Running validation loop...")
    metrics = validate_epoch_exact_copy(model, dataloader, criterion, args.device)
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON: Checkpoint Saved vs Current Validation")
    print("=" * 80)
    
    print(f"\nCheckpoint (saved during training):")
    print(f"  Macro F1: {checkpoint['val_macro_f1']:.2%}")
    if 'val_metrics' in checkpoint:
        saved_metrics = checkpoint['val_metrics']
        print(f"  Accuracy: {saved_metrics['accuracy']:.2%}")
        for cls_name in ['SPLIT', 'RECT', 'AB']:
            if cls_name in saved_metrics['per_class']:
                f1 = saved_metrics['per_class'][cls_name]['f1']
                print(f"    {cls_name:10s} F1: {f1:.2%}")
    
    print(f"\nCurrent validation (checkpoint loaded):")
    print(f"  Macro F1: {metrics['macro_f1']:.2%}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    for cls_name in ['SPLIT', 'RECT', 'AB']:
        f1 = metrics['per_class'][cls_name]['f1']
        print(f"    {cls_name:10s} F1: {f1:.2%}")
    
    # Delta
    delta_f1 = (metrics['macro_f1'] - checkpoint['val_macro_f1']) * 100
    print(f"\n{'✅ MATCH' if abs(delta_f1) < 1.0 else '❌ MISMATCH'}")
    print(f"  Delta F1: {delta_f1:+.2f}pp")
    
    if abs(delta_f1) >= 1.0:
        print("\n⚠️ PROBLEM DETECTED:")
        print("  Checkpoint metrics do NOT match current validation")
        print("  Possible causes:")
        print("    1. BatchNorm running_stats inconsistency")
        print("    2. Checkpoint saved at wrong moment")
        print("    3. Different data loading order (shuffle seed?)")
        print("    4. Model architecture mismatch")


if __name__ == '__main__':
    main()
