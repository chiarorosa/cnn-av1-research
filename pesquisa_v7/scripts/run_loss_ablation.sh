#!/bin/bash
# Run Loss Function Ablation Study (Experiment 04)
# Tests 5 loss functions sequentially

set -e  # Exit on error

# Configuration
DATASET_DIR="pesquisa_v7/v7_dataset/block_16"
STAGE1_CKPT="pesquisa_v7/logs/v7_experiments/solution1_adapter/stage1/stage1_model_best.pt"
OUTPUT_BASE="pesquisa_v7/logs/v7_experiments"
DEVICE="cuda"
BATCH_SIZE=128
EPOCHS=50
SEED=42

echo "========================================"
echo "  Loss Function Ablation Study"
echo "  Experiment 04 - Sequential Execution"
echo "========================================"
echo ""

# Check if Stage 1 checkpoint exists
if [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint not found: $STAGE1_CKPT"
    echo "Please train Stage 1 first using 020_train_adapter_solution.py"
    exit 1
fi

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset not found: $DATASET_DIR"
    echo "Please prepare dataset first using 001_prepare_v7_dataset.py"
    exit 1
fi

echo "Configuration:"
echo "  Dataset: $DATASET_DIR"
echo "  Stage 1: $STAGE1_CKPT"
echo "  Device: $DEVICE"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Seed: $SEED"
echo ""
echo "Will run 5 experiments:"
echo "  1. Baseline (γ=2.0)"
echo "  2. Exp 4A: Focal γ=3.0"
echo "  3. Exp 4B: Poly Loss"
echo "  4. Exp 4C: Asymmetric Loss"
echo "  5. Exp 4D: Focal + Label Smoothing"
echo ""
read -p "Press Enter to start (or Ctrl+C to cancel)..."

# Activate virtual environment
source .venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/home/chiarorosa/CNN_AV1/pesquisa_v7:$PYTHONPATH

# ============================================================================
# BASELINE (for comparison, may skip if already trained)
# ============================================================================
echo ""
echo "========================================"
echo "  BASELINE: ClassBalancedFocalLoss γ=2.0"
echo "========================================"
echo ""

OUTPUT_DIR="${OUTPUT_BASE}/exp04_baseline_focal2"

if [ -f "$OUTPUT_DIR/stage2_adapter/stage2_adapter_model_best.pt" ]; then
    echo "⚠️  Baseline already exists, skipping..."
else
    python3 pesquisa_v7/scripts/021_train_loss_ablation.py \
        --dataset-dir $DATASET_DIR \
        --stage1-checkpoint $STAGE1_CKPT \
        --output-dir $OUTPUT_DIR \
        --loss-type baseline \
        --device $DEVICE \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --seed $SEED
fi

# ============================================================================
# EXP 4A: Focal Loss γ=3.0
# ============================================================================
echo ""
echo "========================================"
echo "  EXP 4A: Focal Loss γ=3.0"
echo "========================================"
echo ""

python3 pesquisa_v7/scripts/021_train_loss_ablation.py \
    --dataset-dir $DATASET_DIR \
    --stage1-checkpoint $STAGE1_CKPT \
    --output-dir ${OUTPUT_BASE}/exp04a_focal_gamma3 \
    --loss-type focal_gamma3 \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --seed $SEED

# ============================================================================
# EXP 4B: Poly Loss
# ============================================================================
echo ""
echo "========================================"
echo "  EXP 4B: Poly Loss (ε=1.0)"
echo "========================================"
echo ""

python3 pesquisa_v7/scripts/021_train_loss_ablation.py \
    --dataset-dir $DATASET_DIR \
    --stage1-checkpoint $STAGE1_CKPT \
    --output-dir ${OUTPUT_BASE}/exp04b_poly_loss \
    --loss-type poly \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --seed $SEED

# ============================================================================
# EXP 4C: Asymmetric Loss
# ============================================================================
echo ""
echo "========================================"
echo "  EXP 4C: Asymmetric Loss"
echo "========================================"
echo ""

python3 pesquisa_v7/scripts/021_train_loss_ablation.py \
    --dataset-dir $DATASET_DIR \
    --stage1-checkpoint $STAGE1_CKPT \
    --output-dir ${OUTPUT_BASE}/exp04c_asymmetric_loss \
    --loss-type asymmetric \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --seed $SEED

# ============================================================================
# EXP 4D: Focal + Label Smoothing
# ============================================================================
echo ""
echo "========================================"
echo "  EXP 4D: Focal + Label Smoothing"
echo "========================================"
echo ""

python3 pesquisa_v7/scripts/021_train_loss_ablation.py \
    --dataset-dir $DATASET_DIR \
    --stage1-checkpoint $STAGE1_CKPT \
    --output-dir ${OUTPUT_BASE}/exp04d_focal_label_smoothing \
    --loss-type focal_smoothing \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --seed $SEED

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================"
echo "  ✓ ALL EXPERIMENTS COMPLETED"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Baseline:   ${OUTPUT_BASE}/exp04_baseline_focal2"
echo "  - Exp 4A:     ${OUTPUT_BASE}/exp04a_focal_gamma3"
echo "  - Exp 4B:     ${OUTPUT_BASE}/exp04b_poly_loss"
echo "  - Exp 4C:     ${OUTPUT_BASE}/exp04c_asymmetric_loss"
echo "  - Exp 4D:     ${OUTPUT_BASE}/exp04d_focal_label_smoothing"
echo ""
echo "Next steps:"
echo "  1. Compare metrics: python3 pesquisa_v7/scripts/022_compare_loss_ablation.py"
echo "  2. Document results: Update docs_v7/04b_resultados_loss_ablation.md"
echo ""
