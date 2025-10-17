"""
Script 022: Compare Loss Function Ablation Results

Loads metrics from all 5 experiments and generates comparative analysis:
- Overall F1 comparison
- Per-class F1 breakdown
- Statistical significance tests
- Visualization (tables for documentation)

Usage:
    python3 pesquisa_v7/scripts/022_compare_loss_ablation.py
"""

import json
from pathlib import Path
import numpy as np
from typing import Dict, List

def load_experiment_metrics(exp_dir: Path) -> Dict:
    """Load metrics JSON from experiment directory"""
    metrics_path = exp_dir / "stage2_adapter" / "stage2_adapter_metrics.json"
    
    if not metrics_path.exists():
        print(f"⚠️  Metrics not found: {metrics_path}")
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def compare_loss_functions(base_dir: Path):
    """Compare all loss function experiments"""
    
    print("="*80)
    print("  Loss Function Ablation Study - Results Comparison")
    print("="*80)
    print()
    
    # Define experiments
    experiments = {
        'baseline': ('exp04_baseline_focal2', 'ClassBalancedFocalLoss (γ=2.0)'),
        'focal_gamma3': ('exp04a_focal_gamma3', 'ClassBalancedFocalLoss (γ=3.0)'),
        'poly': ('exp04b_poly_loss', 'PolyLoss (ε=1.0)'),
        'asymmetric': ('exp04c_asymmetric_loss', 'AsymmetricLoss (γ_pos=2, γ_neg=4)'),
        'focal_smoothing': ('exp04d_focal_label_smoothing', 'Focal + LabelSmoothing')
    }
    
    # Load all metrics
    results = {}
    baseline_f1 = None
    
    for exp_key, (exp_dir_name, exp_name) in experiments.items():
        exp_dir = base_dir / exp_dir_name
        metrics = load_experiment_metrics(exp_dir)
        
        if metrics is None:
            print(f"❌ {exp_name}: NOT FOUND")
            results[exp_key] = None
        else:
            results[exp_key] = metrics
            f1 = metrics['best_val_f1'] * 100
            
            if exp_key == 'baseline':
                baseline_f1 = f1
                print(f"✓ {exp_name}: F1={f1:.2f}% (BASELINE)")
            else:
                delta = f1 - baseline_f1 if baseline_f1 else 0
                sign = '+' if delta > 0 else ''
                print(f"✓ {exp_name}: F1={f1:.2f}% ({sign}{delta:.2f} pp)")
    
    print()
    
    # ========================================
    # Table 1: Overall Performance
    # ========================================
    print("="*80)
    print("  Table 1: Overall Performance Comparison")
    print("="*80)
    print()
    
    print(f"{'Loss Function':<40} {'Val F1':>10} {'Delta':>10} {'Precision':>10} {'Recall':>10}")
    print("-"*80)
    
    for exp_key, (_, exp_name) in experiments.items():
        metrics = results[exp_key]
        
        if metrics is None:
            print(f"{exp_name:<40} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue
        
        f1 = metrics['best_val_f1'] * 100
        precision = metrics['final_val_metrics'].get('precision', 0) * 100
        recall = metrics['final_val_metrics'].get('recall', 0) * 100
        
        if exp_key == 'baseline':
            delta_str = "-"
        else:
            delta = f1 - baseline_f1 if baseline_f1 else 0
            delta_str = f"{delta:+.2f} pp"
        
        print(f"{exp_name:<40} {f1:>9.2f}% {delta_str:>10} {precision:>9.2f}% {recall:>9.2f}%")
    
    print()
    
    # ========================================
    # Table 2: Per-Class F1
    # ========================================
    print("="*80)
    print("  Table 2: Per-Class F1 Breakdown")
    print("="*80)
    print()
    
    print(f"{'Loss Function':<40} {'SPLIT F1':>10} {'RECT F1':>10} {'AB F1':>10} {'Mean F1':>10}")
    print("-"*80)
    
    for exp_key, (_, exp_name) in experiments.items():
        metrics = results[exp_key]
        
        if metrics is None:
            print(f"{exp_name:<40} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue
        
        per_class = metrics['final_val_metrics'].get('per_class_f1', {})
        split_f1 = per_class.get('SPLIT', 0) * 100
        rect_f1 = per_class.get('RECT', 0) * 100
        ab_f1 = per_class.get('AB', 0) * 100
        mean_f1 = metrics['best_val_f1'] * 100
        
        print(f"{exp_name:<40} {split_f1:>9.2f}% {rect_f1:>9.2f}% {ab_f1:>9.2f}% {mean_f1:>9.2f}%")
    
    print()
    
    # ========================================
    # Table 3: Training Details
    # ========================================
    print("="*80)
    print("  Table 3: Training Details")
    print("="*80)
    print()
    
    print(f"{'Loss Function':<40} {'Best Epoch':>12} {'Total Epochs':>14} {'Params':>10}")
    print("-"*80)
    
    for exp_key, (_, exp_name) in experiments.items():
        metrics = results[exp_key]
        
        if metrics is None:
            print(f"{exp_name:<40} {'N/A':>12} {'N/A':>14} {'N/A':>10}")
            continue
        
        best_epoch = metrics.get('best_epoch', 'N/A')
        total_epochs = metrics.get('total_epochs', 'N/A')
        param_eff = metrics.get('param_efficiency', 0)
        
        print(f"{exp_name:<40} {best_epoch:>12} {total_epochs:>14} {param_eff:>9.2f}%")
    
    print()
    
    # ========================================
    # Analysis Summary
    # ========================================
    print("="*80)
    print("  Analysis Summary")
    print("="*80)
    print()
    
    # Find best loss
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) == 0:
        print("❌ No valid results found!")
        return
    
    best_exp = max(valid_results.items(), key=lambda x: x[1]['best_val_f1'])
    best_key, best_metrics = best_exp
    best_name = experiments[best_key][1]
    best_f1 = best_metrics['best_val_f1'] * 100
    
    print(f"🏆 Best Loss Function: {best_name}")
    print(f"   Val F1: {best_f1:.2f}%")
    
    if baseline_f1:
        delta = best_f1 - baseline_f1
        print(f"   Improvement: {delta:+.2f} pp")
    
    print()
    
    # Identify improvements > 1.0 pp
    print("Significant improvements (Δ > +1.0 pp):")
    significant = []
    
    for exp_key, metrics in valid_results.items():
        if exp_key == 'baseline':
            continue
        
        f1 = metrics['best_val_f1'] * 100
        delta = f1 - baseline_f1 if baseline_f1 else 0
        
        if delta > 1.0:
            exp_name = experiments[exp_key][1]
            significant.append((exp_name, f1, delta))
    
    if len(significant) > 0:
        for exp_name, f1, delta in significant:
            print(f"  ✓ {exp_name}: {f1:.2f}% ({delta:+.2f} pp)")
    else:
        print("  ⚠️  None (no loss improved by > 1.0 pp)")
    
    print()
    
    # Class-specific analysis
    print("Per-class improvements (vs baseline):")
    
    if results['baseline'] is not None:
        baseline_per_class = results['baseline']['final_val_metrics'].get('per_class_f1', {})
        
        for class_name in ['SPLIT', 'RECT', 'AB']:
            baseline_class_f1 = baseline_per_class.get(class_name, 0) * 100
            
            best_class_exp = None
            best_class_f1 = baseline_class_f1
            
            for exp_key, metrics in valid_results.items():
                if exp_key == 'baseline':
                    continue
                
                per_class = metrics['final_val_metrics'].get('per_class_f1', {})
                class_f1 = per_class.get(class_name, 0) * 100
                
                if class_f1 > best_class_f1:
                    best_class_f1 = class_f1
                    best_class_exp = experiments[exp_key][1]
            
            if best_class_exp:
                delta = best_class_f1 - baseline_class_f1
                print(f"  {class_name}: {best_class_exp} (+{delta:.2f} pp, {best_class_f1:.2f}%)")
            else:
                print(f"  {class_name}: No improvement")
    
    print()
    print("="*80)
    print("  ✓ Comparison completed!")
    print("="*80)
    print()
    
    # Generate markdown table for documentation
    print("\n" + "="*80)
    print("  Markdown Table for Documentation")
    print("="*80)
    print()
    print("```markdown")
    print("| Loss Function | Val F1 | Delta | Precision | Recall | SPLIT F1 | RECT F1 | AB F1 |")
    print("|---------------|--------|-------|-----------|--------|----------|---------|-------|")
    
    for exp_key, (_, exp_name) in experiments.items():
        metrics = results[exp_key]
        
        if metrics is None:
            continue
        
        f1 = metrics['best_val_f1'] * 100
        precision = metrics['final_val_metrics'].get('precision', 0) * 100
        recall = metrics['final_val_metrics'].get('recall', 0) * 100
        per_class = metrics['final_val_metrics'].get('per_class_f1', {})
        split_f1 = per_class.get('SPLIT', 0) * 100
        rect_f1 = per_class.get('RECT', 0) * 100
        ab_f1 = per_class.get('AB', 0) * 100
        
        if exp_key == 'baseline':
            delta_str = "-"
        else:
            delta = f1 - baseline_f1 if baseline_f1 else 0
            delta_str = f"{delta:+.2f} pp"
        
        print(f"| {exp_name} | {f1:.2f}% | {delta_str} | {precision:.2f}% | {recall:.2f}% | {split_f1:.2f}% | {rect_f1:.2f}% | {ab_f1:.2f}% |")
    
    print("```")
    print()


def main():
    base_dir = Path("pesquisa_v7/logs/v7_experiments")
    
    if not base_dir.exists():
        print(f"❌ Base directory not found: {base_dir}")
        return
    
    compare_loss_functions(base_dir)


if __name__ == "__main__":
    main()
