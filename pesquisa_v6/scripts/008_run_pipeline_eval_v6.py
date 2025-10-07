"""
Script 008: Run Full Pipeline Evaluation (V6)
Evaluates the complete hierarchical pipeline:
- Stage 1: Binary (NONE vs PARTITION)
- Stage 2: 3-way (SPLIT, RECT, AB)
- Stage 3-RECT: Binary (HORZ vs VERT)
- Stage 3-AB: FGVC (HORZ_A, HORZ_B, VERT_A, VERT_B)
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Add v6 pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "v6_pipeline"))
from data_hub import HierarchicalBlockDatasetV6, BlockRecord, build_hierarchical_dataset_v6
from models import Stage1Model, Stage2Model, Stage3RectModel, Stage3ABModel
from metrics import compute_metrics

# Import FGVC model from script 006
# Using importlib to load module by file path for better maintainability
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fgvc_module",
    Path(__file__).parent / "006_train_stage3_ab_fgvc.py"
)
fgvc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fgvc_module)
FGVCModel = fgvc_module.FGVCModel


class HierarchicalPipelineV6:
    """Complete hierarchical pipeline for V6"""
    
    def __init__(self, stage1_model, stage2_model, stage3_rect_model, stage3_ab_model, 
                 stage1_threshold=0.5, device='cuda'):
        self.stage1_model = stage1_model.to(device).eval()
        self.stage2_model = stage2_model.to(device).eval()
        self.stage3_rect_model = stage3_rect_model.to(device).eval()
        self.stage3_ab_model = stage3_ab_model.to(device).eval()
        self.stage1_threshold = stage1_threshold
        self.device = device
        
        # Label mappings (v6)
        self.stage2_to_original = {
            0: 1,  # SPLIT -> SPLIT
            1: 2,  # RECT -> RECT (will be refined)
            2: 3,  # AB -> AB (will be refined)
        }
        
        self.stage3_rect_to_original = {
            0: 2,  # HORZ -> HORZ
            1: 3,  # VERT -> VERT
        }
        
        self.stage3_ab_to_original = {
            0: 4,  # HORZ_A -> HORZ_A
            1: 5,  # HORZ_B -> HORZ_B
            2: 6,  # VERT_A -> VERT_A
            3: 7,  # VERT_B -> VERT_B
        }
    
    @torch.no_grad()
    def predict(self, images):
        """Run full hierarchical prediction"""
        batch_size = images.size(0)
        images = images.to(self.device)
        
        # Stage 1: NONE vs PARTITION
        stage1_logits = self.stage1_model(images)
        stage1_probs = torch.sigmoid(stage1_logits).squeeze()
        stage1_preds = (stage1_probs >= self.stage1_threshold).long()
        
        # Initialize final predictions with NONE (0)
        final_preds = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # Get indices of PARTITION predictions
        partition_mask = stage1_preds == 1
        partition_indices = partition_mask.nonzero(as_tuple=True)[0]
        
        if len(partition_indices) == 0:
            return final_preds.cpu()
        
        # Stage 2: SPLIT, RECT, AB
        partition_images = images[partition_indices]
        stage2_logits = self.stage2_model(partition_images)
        stage2_probs = F.softmax(stage2_logits, dim=1)
        stage2_preds = torch.argmax(stage2_probs, dim=1)
        
        # Process SPLIT (class 0 in Stage 2 -> class 1 in final)
        split_mask = stage2_preds == 0
        split_indices = partition_indices[split_mask]
        final_preds[split_indices] = 1
        
        # Process RECT (class 1 in Stage 2)
        rect_mask = stage2_preds == 1
        rect_indices = partition_indices[rect_mask]
        
        if len(rect_indices) > 0:
            rect_images = images[rect_indices]
            stage3_rect_logits = self.stage3_rect_model(rect_images)
            stage3_rect_probs = F.softmax(stage3_rect_logits, dim=1)
            stage3_rect_preds = torch.argmax(stage3_rect_probs, dim=1)
            
            # Map to final labels: 0->2 (HORZ), 1->3 (VERT)
            final_preds[rect_indices] = stage3_rect_preds + 2
        
        # Process AB (class 2 in Stage 2)
        ab_mask = stage2_preds == 2
        ab_indices = partition_indices[ab_mask]
        
        if len(ab_indices) > 0:
            ab_images = images[ab_indices]
            stage3_ab_logits = self.stage3_ab_model(ab_images)
            stage3_ab_probs = F.softmax(stage3_ab_logits, dim=1)
            stage3_ab_preds = torch.argmax(stage3_ab_probs, dim=1)
            
            # Map to final labels: 0->4, 1->5, 2->6, 3->7
            final_preds[ab_indices] = stage3_ab_preds + 4
        
        return final_preds.cpu()


def evaluate_pipeline(pipeline, dataloader, class_names):
    """Evaluate pipeline on dataset"""
    all_preds = []
    all_labels = []
    
    print("\n  Running pipeline evaluation...")
    for batch in tqdm(dataloader, desc="  Evaluating"):
        images = batch['image']
        labels = batch['label_stage0']  # Ground truth labels (0-7)
        
        preds = pipeline.predict(images)
        
        all_preds.append(preds)
        all_labels.append(labels)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, labels=class_names)
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'metrics': metrics,
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist()
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Full Pipeline Evaluation (V6)")
    parser.add_argument("--dataset-dir", type=str,
                       default="pesquisa_v6/v6_dataset/block_16",
                       help="Dataset directory")
    parser.add_argument("--stage1-model", type=str,
                       default="pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt",
                       help="Stage 1 model path")
    parser.add_argument("--stage2-model", type=str,
                       default="pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt",
                       help="Stage 2 model path")
    parser.add_argument("--stage3-rect-model", type=str,
                       default="pesquisa_v6/logs/v6_experiments/stage3_rect/stage3_rect_model_best.pt",
                       help="Stage 3 RECT model path")
    parser.add_argument("--stage3-ab-models", type=str, nargs=3,
                       default=[
                           "pesquisa_v6/logs/v6_experiments/stage3_ab/stage3_ab_fgvc_best.pt"
                       ],
                       help="Stage 3 AB FGVC model path")
    parser.add_argument("--stage1-threshold", type=float, default=0.45,
                       help="Stage 1 classification threshold (optimized)")
    parser.add_argument("--output-dir", type=str,
                       default="pesquisa_v6/logs/v6_experiments/pipeline_eval",
                       help="Output directory")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--use-test", action="store_true",
                       help="Use test set instead of validation set")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"  V6 Pipeline Evaluation")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Dataset: {args.dataset_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Stage 1 threshold: {args.stage1_threshold}")
    
    # Load models
    print(f"\n[1/4] Loading models...")
    
    # Stage 1
    stage1_model = Stage1Model(pretrained=False)
    checkpoint = torch.load(args.stage1_model, map_location=device, weights_only=False)
    stage1_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✅ Stage 1 loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    
    # Stage 2
    stage2_model = Stage2Model(pretrained=False)
    checkpoint = torch.load(args.stage2_model, map_location=device, weights_only=False)
    stage2_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✅ Stage 2 loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    
    # Stage 3 RECT
    stage3_rect_model = Stage3RectModel(pretrained=False)
    checkpoint = torch.load(args.stage3_rect_model, map_location=device, weights_only=False)
    stage3_rect_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✅ Stage 3-RECT loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    
    # Stage 3 AB FGVC (single model)
    base_ab_model = Stage3ABModel(pretrained=False)
    stage3_ab_model = FGVCModel(base_ab_model, num_classes=4, feat_dim=512)
    checkpoint = torch.load(args.stage3_ab_models[0], map_location=device, weights_only=False)
    stage3_ab_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✅ Stage 3-AB FGVC loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    
    # Create pipeline
    print(f"\n[2/4] Creating pipeline...")
    pipeline = HierarchicalPipelineV6(
        stage1_model, stage2_model, stage3_rect_model, stage3_ab_model,
        stage1_threshold=args.stage1_threshold,
        device=device
    )
    print(f"  ✅ Pipeline created with threshold {args.stage1_threshold}")
    
    # Load dataset
    print(f"\n[3/4] Loading dataset...")
    dataset_dir = Path(args.dataset_dir)
    split = "test" if args.use_test else "val"
    data_file = dataset_dir / f"{split}.pt"
    
    if not data_file.exists():
        print(f"  ⚠️  {split}.pt not found, using val.pt")
        data_file = dataset_dir / "val.pt"
        split = "val"
    
    data = torch.load(data_file, weights_only=False)
    print(f"  Dataset: {split}")
    print(f"  Samples: {len(data['samples'])}")
    
    from augmentation import Stage1Augmentation
    aug = Stage1Augmentation(train=False)
    
    record = BlockRecord(
        samples=data['samples'].numpy().transpose(0, 2, 3, 1),
        labels=data['labels_stage0'].numpy(),
        qps=data['qps'].numpy().reshape(-1, 1)
    )
    dataset = build_hierarchical_dataset_v6(record, augmentation=aug, stage='eval')
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Evaluate
    print(f"\n[4/4] Evaluating pipeline...")
    class_names = ['NONE', 'SPLIT', 'HORZ', 'VERT', 'HORZ_A', 'HORZ_B', 'VERT_A', 'VERT_B']
    
    results = evaluate_pipeline(pipeline, dataloader, class_names)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"  Pipeline Results ({split} set)")
    print(f"{'='*70}")
    print(f"\n  Overall Metrics:")
    print(f"    Accuracy: {results['metrics']['accuracy']:.2%}")
    print(f"    Macro F1: {results['metrics']['macro_f1']:.2%}")
    print(f"    Weighted F1: {results['metrics']['weighted_f1']:.2%}")
    
    print(f"\n  Per-Class F1:")
    for class_name in class_names:
        f1 = results['metrics']['per_class'][class_name]['f1']
        print(f"    {class_name:8s}: {f1:.2%}")
    
    print(f"\n  Classification Report:")
    print(results['classification_report'])
    
    # Save results
    print(f"\n{'='*70}")
    print(f"  Saving Results")
    print(f"{'='*70}")
    
    # Save detailed metrics
    metrics_file = output_dir / f"pipeline_metrics_{split}.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'split': split,
            'threshold': args.stage1_threshold,
            'metrics': results['metrics'],
            'confusion_matrix': results['confusion_matrix'],
            'class_names': class_names,
            'config': vars(args)
        }, f, indent=2)
    print(f"  ✅ Metrics saved: {metrics_file}")
    
    # Save predictions
    preds_file = output_dir / f"pipeline_predictions_{split}.npz"
    np.savez(preds_file, 
             predictions=results['predictions'],
             labels=results['labels'],
             class_names=class_names)
    print(f"  ✅ Predictions saved: {preds_file}")
    
    # Save report
    report_file = output_dir / f"pipeline_report_{split}.txt"
    with open(report_file, 'w') as f:
        f.write(f"V6 Pipeline Evaluation Report\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Dataset: {split}\n")
        f.write(f"Stage 1 Threshold: {args.stage1_threshold}\n")
        f.write(f"Samples: {len(results['labels'])}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy: {results['metrics']['accuracy']:.2%}\n")
        f.write(f"  Macro F1: {results['metrics']['macro_f1']:.2%}\n")
        f.write(f"  Weighted F1: {results['metrics']['weighted_f1']:.2%}\n\n")
        f.write(f"Classification Report:\n")
        f.write(results['classification_report'])
    print(f"  ✅ Report saved: {report_file}")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
