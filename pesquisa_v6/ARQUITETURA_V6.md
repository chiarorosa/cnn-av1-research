# Arquitetura Pipeline v6 - Diagrama Visual

## 🏗️ Visão Geral da Hierarquia

```
                          ┌─────────────────────────────┐
                          │     Input Block (16x16)     │
                          │   3 channels (Y, U, V)      │
                          └──────────────┬──────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │        Backbone CNN         │
                          │  ResNet-18 + SE-Block       │
                          │  + Spatial Attention        │
                          │  + Dropout Progressivo      │
                          │  Output: 512-dim features   │
                          └──────────────┬──────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │         STAGE 1: Binary Head            │
                    │      FC[512→256→1] + Sigmoid            │
                    │   Focal Loss (γ=2.5) + Hard Mining      │
                    │   Threshold: 0.55 (optimized)           │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────┐
                    │  P(PARTITION) >= threshold?      │
                    └────┬─────────────────────┬───────┘
                         │ NO                  │ YES
                         ▼                     ▼
                ┌────────────────┐   ┌─────────────────────────┐
                │ PARTITION_NONE │   │    STAGE 2: 3-Way Head  │
                │  (Final Output)│   │   FC[512→256→128→3]     │
                └────────────────┘   │  Classes: SPLIT|RECT|AB │
                                     │  CB-Focal (β=0.9999)    │
                                     └───┬────────┬────────┬───┘
                                         │        │        │
                        ┌────────────────▼──┐  ┌──▼────┐  │
                        │     SPLIT         │  │ RECT  │  │
                        │ PARTITION_SPLIT   │  │       │  │
                        │  (Final Output)   │  │       │  │
                        └───────────────────┘  └───┬───┘  │
                                                   │      │
                                    ┌──────────────▼─┐    │
                                    │ STAGE 3-RECT   │    │
                                    │  Binary Head   │    │
                                    │  HORZ vs VERT  │    │
                                    │ FC[512→128→2]  │    │
                                    └──┬──────────┬──┘    │
                                       │          │       │
                              ┌────────▼─┐   ┌───▼──────┐│
                              │ PARTITION│   │PARTITION ││
                              │   _HORZ  │   │  _VERT   ││
                              └──────────┘   └──────────┘│
                                                          │
                                          ┌───────────────▼──────────────┐
                                          │      STAGE 3-AB: Ensemble    │
                                          │                              │
                                          │  ┌──────────┐ ┌──────────┐  │
                                          │  │ Model 1  │ │ Model 2  │  │
                                          │  │  4-way   │ │  4-way   │  │
                                          │  └────┬─────┘ └─────┬────┘  │
                                          │       │   ┌──────┐  │       │
                                          │       └───▶│Model│◀─┘       │
                                          │           │  3  │           │
                                          │           └──┬──┘           │
                                          │              │               │
                                          │      ┌───────▼────────┐     │
                                          │      │ Majority Vote  │     │
                                          │      └───────┬────────┘     │
                                          └──────────────┼──────────────┘
                                                         │
                        ┌────────────────────────────────┴─────────────────────────┐
                        │                                                          │
               ┌────────▼─────────┐  ┌────────────────┐  ┌───────────────┐  ┌────▼──────────┐
               │ PARTITION_HORZ_A │  │PARTITION_HORZ_B│  │PARTITION_VERT_A│  │PARTITION_VERT_B│
               │  (Final Output)  │  │ (Final Output) │  │ (Final Output) │  │ (Final Output) │
               └──────────────────┘  └────────────────┘  └───────────────┘  └───────────────┘
```

## 📊 Comparação v5 vs v6

### Stage 2 Redesign

**v5 (5 classes):**
```
STAGE 2 → [NONE, SPLIT, RECT, AB, 1TO4]
          │
          ├─ NONE: 0 samples (já filtrado)
          ├─ SPLIT: 5,962 samples
          ├─ RECT: 17,765 samples
          ├─ AB: 14,529 samples
          └─ 1TO4: 0 samples (não existe)

Problemas:
- 2 classes vazias (NONE, 1TO4)
- Confusão RECT ↔ AB: 8,134 erros
- Macro F1: 33.41%
```

**v6 (3 classes):**
```
STAGE 2 → [SPLIT, RECT, AB]
          │
          ├─ SPLIT: 5,962 samples (15.6%)
          ├─ RECT: 17,765 samples (46.4%)
          └─ AB: 14,529 samples (38.0%)

Melhorias:
- Remove classes vazias
- Foco nas 3 macro-classes relevantes
- Melhor separabilidade
- Target Macro F1: 48%+
```

### Stage 3-AB: Single Model vs Ensemble

**v5 (Single Model):**
```
                  ┌──────────────────┐
                  │   AB Classifier  │
                  │   4-way (LSTM?)  │
                  └────────┬─────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     Predições:       Problema:         Real:
     VERT_B: 44%      Colapso!          Balanced
     HORZ_A: 28%      para VERT_B       ~25% each
     VERT_A: 18%
     HORZ_B: 10%

F1: 25.26% (CRÍTICO!)
```

**v6 (Ensemble de 3):**
```
                  ┌──────────────────┐
                  │   Model 1 (4-way)│
                  │  Heavy Aug Set 1 │
                  └────────┬─────────┘
                           │
                  ┌────────▼─────────┐
                  │   Model 2 (4-way)│
                  │  Heavy Aug Set 2 │
                  └────────┬─────────┘
                           │
                  ┌────────▼─────────┐
                  │   Model 3 (4-way)│
                  │  Heavy Aug Set 3 │
                  └────────┬─────────┘
                           │
                  ┌────────▼─────────┐
                  │  Majority Vote   │
                  │  (3 predictions) │
                  └──────────────────┘

Melhorias:
- 3 modelos independentes
- Aug diferente por modelo
- Votação robusta
- Target F1: 50%+
```

## 🔧 Componentes Técnicos Detalhados

### 1. Backbone Architecture

```
Input (16x16x3)
    ↓
Conv1 (7x7, 64, stride 2) + BN + ReLU
    ↓
MaxPool (3x3, stride 2)
    ↓
ResBlock 1 (64, 64) × 2
    ↓
SE-Block (ratio=16)          ← Attention
    ↓
Dropout (p=0.1)              ← Regularization
    ↓
ResBlock 2 (64, 128) × 2
    ↓
SE-Block (ratio=16)
    ↓
Dropout (p=0.2)
    ↓
ResBlock 3 (128, 256) × 2
    ↓
SE-Block (ratio=16)
    ↓
Dropout (p=0.3)
    ↓
ResBlock 4 (256, 512) × 2
    ↓
SE-Block (ratio=16)
    ↓
Dropout (p=0.4)
    ↓
Spatial Attention Module     ← Attention
    ↓
Global Average Pooling
    ↓
Features (512-dim)
```

### 2. SE-Block (Squeeze-and-Excitation)

```
Input (C channels)
    ↓
Global Average Pool → (C,)
    ↓
FC(C → C/16) + ReLU
    ↓
FC(C/16 → C) + Sigmoid → weights (C,)
    ↓
Channel-wise multiply
    ↓
Output (C channels, reweighted)
```

### 3. Spatial Attention Module

```
Input (H×W×C)
    ↓
MaxPool(channel) → (H×W×1)
AvgPool(channel) → (H×W×1)
    ↓
Concat → (H×W×2)
    ↓
Conv(7×7) + Sigmoid → attention map (H×W×1)
    ↓
Element-wise multiply
    ↓
Output (H×W×C, spatially reweighted)
```

### 4. Loss Functions

**Stage 1: Focal Loss + Hard Negative Mining**
```python
# Focal Loss
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
# γ = 2.5 (mais agressivo que v5)
# α = [0.25, 0.75] (balancear NONE vs PARTITION)

# Hard Negative Mining
1. Forward pass
2. Calcular loss por sample
3. Ordenar losses (maior → menor)
4. Manter top-k negatives (ratio 3:1)
5. Backward apenas nesses samples
```

**Stage 2: Class-Balanced Focal Loss**
```python
# Effective Number of Samples
E_n = (1 - β^n) / (1 - β)
# β = 0.9999 (favorece classes raras)

# Class Weights
w_i = E_total / E_i

# CB Focal Loss
CB_FL = w_i * FL(p_i)
```

**Stage 3-AB: Mixup + Focal Loss**
```python
# Mixup
λ ~ Beta(α, α)  # α = 0.4
x_mix = λ * x_i + (1-λ) * x_j
y_mix = λ * y_i + (1-λ) * y_j

# Loss
L = λ * FL(p, y_i) + (1-λ) * FL(p, y_j)
```

### 5. Data Augmentation Pipeline

**Stage 3-AB Heavy Augmentation:**
```python
if horizontal_flip (p=0.5):
    image = flip_horizontal(image)
    # Swap labels
    if label == HORZ_A: label = HORZ_B
    if label == HORZ_B: label = HORZ_A
    if label == VERT_A: label = VERT_B
    if label == VERT_B: label = VERT_A

if vertical_flip (p=0.5):
    image = flip_vertical(image)
    # Similar swap

if rotate_90 (p=0.25):
    image = rotate(image, 90)
    # Rotate labels: HORZ ↔ VERT
    if label in [HORZ_A, HORZ_B]:
        label = corresponding_VERT
    elif label in [VERT_A, VERT_B]:
        label = corresponding_HORZ

if mixup (p=0.3):
    λ = sample_beta(0.4, 0.4)
    image = λ * image + (1-λ) * other_image
    # Soft label

if coarse_dropout (p=0.3):
    # Drop 4-8 patches of size 4x4
    image = coarse_dropout(image, n_holes=4-8, size=4)
```

## 📈 Fluxo de Treino

### Phase 1: Backbone Pre-training (Stage 1)

```
Epochs: 20
Batch Size: 128
Optimizer: AdamW (lr=1e-3, wd=1e-4)
Scheduler: CosineAnnealingLR (1e-3 → 1e-5)
Loss: Focal (γ=2.5) + Hard Mining (3:1)
Sampling: Balanced (1:1 NONE vs PARTITION)
Augmentation: Basic (Flip, Rotate, Noise)

Freeze: None
Train: Backbone + Stage1 Head
```

### Phase 2: Stage 2 Fine-tuning

```
Epochs: 25
  - Epochs 1-2: Freeze Backbone
  - Epochs 3-25: Unfreeze Backbone (lr=1e-5)

Batch Size: 128
Optimizer: AdamW
  - Stage2 Head: lr=5e-4 → 5e-6
  - Backbone: lr=1e-5 (fixed)

Loss: CB-Focal (β=0.9999, γ=2.0)
Sampling: Effective Number (β=0.9999)
Augmentation: Heavy (Flip, Rotate, Cutout, GridShuffle)

Checkpoint: Load from Phase 1
```

### Phase 3: Specialists (Stage 3)

**RECT Specialist:**
```
Epochs: 30
Batch Size: 128
Optimizer: AdamW (lr=3e-4, wd=1e-4)
Loss: Focal (γ=2.0)
Augmentation: Medium

Freeze: Backbone + Stage1 + Stage2
Train: RECT Head only
```

**AB Ensemble (3 models):**
```
For each model (i=1,2,3):
    Epochs: 50
    Batch Size: 64 (menor devido a Mixup)
    Optimizer: AdamW (lr=3e-4, wd=1e-4)
    Loss: Mixup (α=0.4) + Focal (γ=2.0)
    Augmentation: Heavy (seed=i, diferente por modelo)
    Sampling: Oversample 5x (minoritárias)
    
    Freeze: Backbone + Stage1 + Stage2
    Train: AB Head only
    Seed: Different random seed per model
```

## 🎯 Métricas de Sucesso por Fase

### Fase 1: Stage 1
```
Baseline (v5):     F1=65.19%, Prec=53.71%, Rec=82.93%
Target (v6 P1):    F1=68-70%, Prec=62-65%, Rec=78-82%
Target (v6 P2):    F1=72-75%, Prec=68-72%, Rec=75-80%

Strategy: Trocar recall por precisão
→ Reduzir falsos positivos de 27k para <15k
```

### Fase 2: Stage 2
```
Baseline (v5):     Macro F1=33.41%
Target (v6 P1):    Macro F1=45-50%
Target (v6 P2):    Macro F1=55-60%

Strategy: Simplificar classes (5→3)
→ Reduzir confusão RECT↔AB de 8k para <4k
```

### Fase 3: Stage 3-AB
```
Baseline (v5):     F1=25.26% (1 model)
Target (v6 P1):    F1=45-50% (ensemble)
Target (v6 P2):    F1=60-65% (ensemble)

Strategy: Ensemble + Heavy Aug + Oversampling
→ Eliminar colapso para VERT_B (de 44% para ~25%)
```

### Final: Pipeline
```
Baseline (v5):     Accuracy=39.56%
Target (v6 P1):    Accuracy=48-52%
Target (v6 P2):    Accuracy=58-63%

Strategy: Melhorias cumulativas em cada stage
```

---

**Última Atualização**: 2025-10-06  
**Status**: 🚧 Arquitetura Definida
