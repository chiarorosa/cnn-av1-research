# CNN-AV1 Research Project - AI Agent Instructions

## Project Overview

**⚠️ CRITICAL: This is a PhD-level technical-scientific research project.**

This doctoral research develops deep learning methods to predict **AV1 video codec partition decisions** using hierarchical CNNs. The system analyzes 16×16 pixel blocks (YUV 4:2:0 10-bit format) and predicts which of 10 partition types the AV1 encoder would choose.

**Research Goal:** Replace expensive encoder partition search with fast CNN inference to accelerate AV1 video encoding.

**Academic Context:**
- **Level:** PhD (Doutorado) research - requires rigorous scientific methodology
- **Standards:** All architectural and algorithmic decisions MUST be grounded in peer-reviewed literature
- **Documentation:** Every experiment, hypothesis, and design choice must be traceable to academic sources
- **Contribution:** Novel solutions to negative transfer in hierarchical video codec prediction

**Agent Requirements:**
1. **Literature-Based Decisions:** Cite papers (author, year) for every architectural choice
2. **Scientific Rigor:** Formulate hypotheses, design controlled experiments, analyze results quantitatively
3. **PhD-Level Creativity:** Propose novel solutions inspired by state-of-the-art (transformers, adapters, meta-learning, etc.)
4. **Reproducibility:** Document all experiments with protocols, hyperparameters, seeds, artifacts
5. **Critical Analysis:** Question assumptions, identify limitations, compare against baselines
6. **NO TIME ESTIMATES:** Never specify dates, deadlines, hours, or days of effort. Focus on technical steps and decisions only.

## Architecture Philosophy

### Three-Stage Hierarchical Pipeline (v6)

The project uses a **coarse-to-fine** classification hierarchy to handle class imbalance:

```
Stage 1 (Binary): NONE vs PARTITION (any split)
  ├─ If NONE → output PARTITION_NONE
  └─ If PARTITION → Stage 2
      
Stage 2 (3-way): SPLIT vs RECT vs AB (macro-classes)
  ├─ SPLIT → output PARTITION_SPLIT
  ├─ RECT → Stage 3-RECT (binary: HORZ vs VERT)
  └─ AB → Stage 3-AB (4-way ensemble: HORZ_A/B, VERT_A/B)
```

**Why hierarchical?** Direct 10-class classification fails due to extreme imbalance. The hierarchy progressively narrows decisions, with specialized heads trained on balanced subsets.

### Versioning: v5 vs v6

- **`pesquisa_v5/`**: Legacy pipeline with 5-class Stage 2 (includes NONE and 1TO4). Has documented issues with catastrophic forgetting and Stage 3-AB collapse (F1=25%).
- **`pesquisa_v6/`**: Current research focus. Redesigned Stage 2 (3 classes only), ensemble-based Stage 3-AB, improved attention mechanisms. **All new development happens here.**

## Data Format and Processing

### AV1 Partition Types (10 classes)

```python
0: PARTITION_NONE      # No split
1: PARTITION_HORZ      # Horizontal split (2 blocks)
2: PARTITION_VERT      # Vertical split (2 blocks)
3: PARTITION_SPLIT     # Quad split (4 blocks)
4: PARTITION_HORZ_A    # Asymmetric horizontal (top)
5: PARTITION_HORZ_B    # Asymmetric horizontal (bottom)
6: PARTITION_VERT_A    # Asymmetric vertical (left)
7: PARTITION_VERT_B    # Asymmetric vertical (right)
8: PARTITION_HORZ_4    # Horizontal 4-way (rarely used)
9: PARTITION_VERT_4    # Vertical 4-way (rarely used)
```

### Raw Data Pipeline (v5 scripts 004-007)

**Critical:** Maintains **lossless 10-bit** precision throughout:

1. **Script 004**: Parse AV1 encoder partition logs → Excel files per frame
2. **Script 005**: Extract Y-component blocks from YUV 4:2:0 10-bit videos → binary files (`uint16` little-endian)
3. **Script 006**: Merge block files by sequence
4. **Script 007**: Generate aligned label and QP files

Data directory structure: `/home/chiarorosa/experimentos/uvg/`
```
intra_raw_blocks/  # Binary block data (10-bit uint16)
labels/            # Partition type labels
qps/               # Quantization parameter values
```

### Dataset Preparation (v6 scripts 001-002)

- **001**: Creates main train/val split from raw data → `v6_dataset/block_16/{train,val}.pt`
- **002**: Creates Stage 3 specialist datasets (RECT, AB with 3 ensemble variants) → `v6_dataset_stage3/`

**Always** check `v6_dataset/block_16/metadata.json` for class distributions before training.

## Model Architecture Components

### Backbone: ResNet-18 + Attention

All models share a common feature extractor (`ImprovedBackbone` in `v6_pipeline/models.py`):

- **Base:** ResNet-18 pretrained on ImageNet (first conv modified for 1-channel grayscale)
- **SE-Blocks** (Squeeze-and-Excitation): Channel attention after each residual layer
- **Spatial Attention Module:** CBAM-style spatial attention before global pooling
- **Progressive Dropout:** 0.1 → 0.2 → 0.3 → 0.4 across layers
- **Output:** 512-dim feature vector

### Stage-Specific Heads

1. **Stage1BinaryHead**: Binary classifier with Temperature Scaling for calibration
2. **Stage2Model**: 3-way classifier with Class-Balanced Focal Loss
3. **Stage3RectModel**: Binary HORZ vs VERT classifier
4. **Stage3ABModel**: 4-way classifier (HORZ_A/B, VERT_A/B) with FGVC techniques
5. **ABEnsemble**: Majority voting over 3 independently trained Stage3AB models

## Loss Functions (`v6_pipeline/losses.py`)

- **FocalLoss**: For binary/multi-class imbalance (γ=2.0-2.5, α=0.25)
- **ClassBalancedFocalLoss**: Uses effective number of samples (β=0.9999)
- **HardNegativeMiningLoss**: For Stage 1 binary classification
- **MixupLoss**: Data augmentation for training stability

**Key Pattern:** Always use `register_buffer()` for class weights to ensure they move with `model.to(device)`.

## Training Workflow (v6)

### Standard Training Pattern

All training scripts follow this structure:

```python
# 1. Load dataset from .pt files
dataset = HierarchicalBlockDatasetV6(...)

# 2. Create balanced sampler if needed
sampler = create_balanced_sampler(dataset, ...)

# 3. Build model with proper loss
model = StageXModel(pretrained=True)
criterion = FocalLoss(gamma=2.0, alpha=0.25)

# 4. Optimizer with discriminative LRs
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': lr_backbone},
    {'params': model.head.parameters(), 'lr': lr_head}
])

# 5. Training loop with early stopping
best_f1 = 0
for epoch in range(epochs):
    train_metrics = train_epoch(...)
    val_metrics = validate_epoch(...)
    if val_metrics['f1'] > best_f1:
        save_checkpoint(model, 'best')
```

### Script Execution Order

```bash
# 1. Prepare datasets
python3 pesquisa_v6/scripts/001_prepare_v6_dataset.py --base-path <data_path>
python3 pesquisa_v6/scripts/002_prepare_v6_stage3_datasets.py --base-path <data_path>

# 2. Train stages sequentially (each depends on previous)
python3 pesquisa_v6/scripts/003_train_stage1_improved.py --dataset-dir v6_dataset/block_16
python3 pesquisa_v6/scripts/004_train_stage2_redesigned.py --stage1-model <ckpt>
python3 pesquisa_v6/scripts/005_train_stage3_rect.py --stage2-model <ckpt>
python3 pesquisa_v6/scripts/006_train_stage3_ab_fgvc.py --stage2-model <ckpt>

# 3. Optimize thresholds
python3 pesquisa_v6/scripts/007_optimize_thresholds.py --model-path <stage1_ckpt>

# 4. Evaluate full pipeline
python3 pesquisa_v6/scripts/008_run_pipeline_eval_v6.py --stage1-model <ckpt> ...
```

**Critical:** Each stage depends on the checkpoint from the previous stage. Check `logs/v6_experiments/*/` for saved models.

## Known Issues and Research Challenges

### Catastrophic Forgetting in Stage 2

**Problem:** Fine-tuning the backbone after frozen training causes F1 to drop from 46% → 32%.

**Root Cause:** Negative Transfer - Stage 1 (binary) and Stage 2 (3-way) features are incompatible. Documented in `PLANO_v6_val2.md` with 5 attempted solutions (ULMFiT, discriminative LR, etc.) that **all failed**.

**Current Workaround:** Save model at best frozen epoch (usually epoch 1-8) before unfreezing.

### Stage 3-AB Class Imbalance

**Problem:** The AB specialist (4-way) has extreme imbalance:
- VERT_A: ~25%
- VERT_B: ~35% 
- HORZ_A: ~20%
- HORZ_B: ~20%

**Solution (v6):** Fine-Grained Visual Classification (FGVC) techniques in `006_train_stage3_ab_fgvc.py`:
- Center Loss for intra-class compactness
- CutMix augmentation
- Cosine classifier with temperature scaling
- Two-stage fine-tuning (freeze → unfreeze)

**Result:** F1=24.5% (4/4 classes predicted) vs v5's 25.26% (1 class collapsed).

## Development Conventions

### File Organization

- **`pesquisa_v{N}/scripts/`**: Numbered training/evaluation scripts (001-009)
- **`pesquisa_v{N}/v{N}_pipeline/`**: Reusable Python modules (models, losses, data_hub, etc.)
- **`pesquisa_v{N}/logs/`**: Training artifacts (checkpoints, metrics, history)
- **`pesquisa_v{N}/notebooks/`**: Jupyter notebooks for analysis (not for training)
- **`pesquisa_v{N}/docs/`**: Markdown documentation (reproducibility guides)
- **`pesquisa_v{N}/docs_v{N}/`**: **PhD thesis documentation** (experiments, analysis, literature reviews)

### Checkpoint Naming Convention

```
stage{N}_model_best.pt        # Best validation F1
stage{N}_model_final.pt       # Last epoch
stage{N}_history.pt           # Training history (losses, metrics per epoch)
stage{N}_metrics.json         # Final metrics summary
```

### Metrics Tracking

All training scripts save:
1. **Checkpoints** (.pt): Model state_dict
2. **History** (.pt): Per-epoch losses and metrics
3. **Metrics JSON**: Final test set performance with per-class breakdown

**Key Metrics:** F1-score (macro for multi-class), Precision, Recall, Accuracy

## Quick Reference Commands

```bash
# Activate environment
source .venv/bin/activate

# Check dataset statistics
python3 -c "import torch; d=torch.load('pesquisa_v6/v6_dataset/block_16/metadata.json'); print(d)"

# Run Stage 1 training with default hyperparameters
python3 pesquisa_v6/scripts/003_train_stage1_improved.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --output-dir pesquisa_v6/logs/v6_experiments/stage1

# Evaluate pipeline (requires all stage checkpoints)
python3 pesquisa_v6/scripts/008_run_pipeline_eval_v6.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --stage1-model pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt \
  --stage2-model pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt \
  --stage3-rect-model pesquisa_v6/logs/v6_experiments/stage3_rect/stage3_rect_model_best.pt \
  --stage3-ab-models <model1> <model2> <model3>  # Ensemble requires 3 models
```

## Documentation Hierarchy

1. **High-level architecture:** `pesquisa_v6/ARQUITETURA_V6.md` (visual diagrams)
2. **Research plan & status:** `pesquisa_v6/PLANO_V6.md`, `PLANO_v6_val2.md`
3. **User guide:** `pesquisa_v6/README.md` (all scripts + parameters)
4. **Reproducibility:** `pesquisa_v5/docs/reproducibility_full_pipeline.md` (v5 only)
5. **PhD thesis materials:** `pesquisa_v6/docs_v6/` (technical-scientific documentation)
6. **Notebooks:** `pesquisa_v6/notebooks/` for result analysis

**When making changes:** 
- Update `README.md` with new script parameters
- Update `CHANGELOG` with architectural modifications
- Document experiments in `docs_v6/` with full scientific rigor (hypothesis, protocol, results, analysis)
- Cite relevant papers for all design decisions

## PhD-Level Expectations

### When Proposing Solutions

**Always include:**
1. **Literature Review:** What papers address similar problems? (cite 3-5 papers)
2. **Hypothesis:** What do you expect to happen and why?
3. **Theoretical Foundation:** Explain the mechanism (e.g., "attention allows model to focus on...")
4. **Expected Results:** Quantitative predictions (e.g., "F1 should increase by 5-10%")
5. **Validation Plan:** How will you confirm success or failure?

**Example (Good):**
```
Problem: Stage 2 AB class has low F1=10.94%

Proposed Solution: Add Spatial Attention Module
- Literature: CBAM (Woo et al., 2018) improved fine-grained classification by 2-3%
- Hypothesis: AB partitions require spatial reasoning (asymmetric borders). 
  Attention will highlight relevant regions.
- Mechanism: Channel attention → spatial attention → refined features
- Expected: AB F1 10.94% → 15-18% (similar to CBAM gains on CUB-200)
- Validation: Train with/without attention, compare AB F1 and attention maps
```

**Example (Bad):**
```
Problem: AB F1 is low

Solution: Add attention

Why: Attention helps models learn better
```

### When Analyzing Results

**Always include:**
1. **Quantitative Comparison:** Tables with metrics (F1, precision, recall per-class)
2. **Statistical Significance:** Did the change matter? (>5% improvement = meaningful)
3. **Error Analysis:** Which samples failed? Why?
4. **Ablation Study:** What happens if you remove component X?
5. **Literature Comparison:** How do your results compare to cited papers?

### When Documenting Experiments

**Create structured documents in `docs_v6/` with:**
1. **Motivation:** Why this experiment? (2-3 paragraphs)
2. **Literature Foundation:** Papers that inspired the approach (5-10 citations)
3. **Hypothesis:** Clear testable statement
4. **Protocol:** Full reproducibility (code, hyperparameters, data splits)
5. **Results:** Tables, plots, per-class breakdown
6. **Analysis:** Why did it work/fail? What does this tell us?
7. **Limitations:** What couldn't be tested? Why?
8. **Next Steps:** What should be tried next?
9. **Artifacts:** Checkpoint paths, logs, notebooks
10. **References:** Full bibliography

## Testing Philosophy

This is research code, not production software:
- **No unit tests** - validation happens through training metrics
- **Scripts are self-contained** - each can run independently with proper arguments
- **Verification:** Check F1 scores match expected ranges from documentation
- **Reproducibility:** Use fixed seeds (`--seed 42`) and document hyperparameters in script docstrings

## Common Debugging Patterns

1. **CUDA out of memory:** Reduce `--batch-size` (128 → 64 → 32)
2. **Low F1 in Stage 2/3:** Check class distribution in dataset metadata - may need rebalancing
3. **NaN losses:** Usually learning rate too high - reduce by 10x
4. **Model not loading:** Check backbone parameter names match (v5 vs v6 have different structures)
5. **Pipeline evaluation fails:** Verify all checkpoint paths exist and were trained on same dataset split

## External Dependencies

- Raw data: `/home/chiarorosa/experimentos/uvg/` (YUV videos + partition logs)
- Video sources: `/home/chiarorosa/videoset/ugc-yuv/` (.y4m files)
- Python env: `.venv` with PyTorch, torchvision, numpy, pandas, openpyxl, scikit-learn

**GPU Required:** All training scripts default to `--device cuda`. Can override to `cpu` but expect 10-50x slowdown.
