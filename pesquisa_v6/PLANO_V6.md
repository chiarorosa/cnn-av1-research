# Pipeline v6 - Plano de Desenvolvimento

## 📋 Sumário Executivo

O pipeline v6 é uma reformulação completa da arquitetura hierárquica, focando em resolver os **3 problemas críticos** identificados na v5:

1. **Stage 1**: Baixa precisão (53.71%) → 27k falsos positivos contaminando o pipeline
2. **Stage 2**: Confusão entre macro-classes (33.41% Macro F1) → Erros propagados
3. **Stage 3-AB**: Colapso total do especialista (25.26% F1) → Classes perdidas

---

## 🎯 Objetivos e Metas

### Métricas Alvo (block_16)

| Componente | v5 Atual | v6 Meta Fase 1 | v6 M3. **Implementar v6_pipeline module** (base code)
4. **Preparar datasets v6** (script 001)
5. **Executar Fase 1** (Experimentos 1.1 e 1.2)
6. **Avaliar resultados** e ajustar plano
7. **Iterar** até atingir critérios de sucesso

---

**Status**: 🟡 Em Desenvolvimento (Training Scripts 6/7 - Script 008 validado)  
**Próximo**: Script 009 (compare_v5_v6) - ÚLTIMO SCRIPT  
**Última Atualização**: 2025-10-06  
**Responsável**: @chiarorosatus**: 🟢 Core Modules Completos | Training Scripts 3/7  
**Última Atualização**: 2025-10-06  
**Responsável**: @chiarorosa  
**Próximo**: Implementar 006_train_stage3_ab_ensemble.pye 2 |
|------------|----------|----------------|----------------|
| **Stage 1 F1** | 65.19% | 68-70% | 72-75% |
| **Stage 1 Precisão** | 53.71% | 62-65% | 68-72% |
| **Stage 2 Macro F1** | 33.41% | 45-50% | 55-60% |
| **Stage 3-RECT F1** | 72.50% | 75-78% | 80-83% |
| **Stage 3-AB F1** | 25.26% | 45-50% | 60-65% |
| **Acurácia Final** | 39.56% | 48-52% | 58-63% |

---

## 🏗️ Arquitetura Proposta

### Estratégia Principal: Hierarquia Revisada (3 Estágios Otimizados)

```
                    ┌─────────────────┐
                    │  Input Block    │
                    │    (16x16)      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Backbone      │
                    │  (ResNet-18+)   │
                    │   + Attention   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────────┐
                    │   STAGE 1 (Binary)  │
                    │  NONE vs PARTITION  │
                    │   + Focal Loss      │
                    │   + Threshold Opt   │
                    └──────┬──────────────┘
                           │
                    ┌──────▼──────┐
                    │   IF NONE   │──→ OUTPUT: PARTITION_NONE
                    └─────────────┘
                           │
                    ┌──────▼─────────────┐
                    │   STAGE 2 (3-way)  │
                    │  SPLIT | RECT | AB │
                    │  + Class Weights   │
                    │  + Data Aug        │
                    └──┬────────┬────┬───┘
                       │        │    │
              ┌────────▼──┐  ┌──▼────▼─────────┐
              │  SPLIT    │  │  STAGE 3-RECT   │
              │  (direct) │  │  HORZ vs VERT   │
              └───────────┘  └─────────────────┘
                                      │
                             ┌────────▼─────────────┐
                             │   STAGE 3-AB         │
                             │  HORZ_A | HORZ_B |   │
                             │  VERT_A | VERT_B     │
                             │  + Ensemble (3 nets) │
                             │  + Heavy Aug         │
                             └──────────────────────┘
```

### Mudanças Chave vs v5

1. **Stage 2 Simplificado**: 
   - ❌ Remove classe "NONE" (já filtrada em Stage 1)
   - ❌ Remove classe "1TO4" (não existe no dataset)
   - ✅ Apenas 3 classes: SPLIT, RECT, AB
   - ✅ Reduz confusão e melhora separabilidade

2. **Stage 1 Melhorado**:
   - ✅ Focal Loss com γ=2.5 (mais agressivo)
   - ✅ Hard Negative Mining
   - ✅ Threshold otimizado via validação (testar 0.45-0.6)
   - ✅ Calibração de probabilidades (Temperature Scaling)

3. **Stage 3-AB Redesenhado**:
   - ✅ Ensemble de 3 modelos (votação majoritária)
   - ✅ Data Augmentation pesado (rotação, flip, cutout)
   - ✅ Oversampling agressivo (5x para classes minoritárias)
   - ✅ Mixup entre classes AB

---

## 🔧 Melhorias Técnicas Implementadas

### 1. Arquitetura do Modelo

#### Backbone Upgrade
```python
# v5: ResNet-18 simples
# v6: ResNet-18 + SE-Block + Dropout adaptativo

- ResNet-18 base
- Squeeze-and-Excitation blocks (channel attention)
- Spatial Attention Module (SAM)
- Dropout progressivo: [0.1, 0.2, 0.3, 0.4]
- Group Normalization (alternativa a BatchNorm)
```

#### Heads Especializados
```python
# Stage 1: Binary Head
- FC [512 → 256 → 1]
- Dropout 0.3
- Sigmoid + Temperature Scaling

# Stage 2: 3-way Head  
- FC [512 → 256 → 128 → 3]
- Dropout 0.4
- Softmax + Label Smoothing (0.1)

# Stage 3-RECT: Binary Head
- FC [512 → 128 → 64 → 2]
- Dropout 0.2

# Stage 3-AB: 4-way Ensemble
- 3x FC [512 → 256 → 128 → 4]
- Dropout 0.5
- Votação majoritária
```

### 2. Loss Functions

#### Stage 1: Focal Loss + Hard Negative Mining
```python
FocalLoss(alpha=0.25, gamma=2.5) 
+ HardNegativeMining(ratio=3:1)
```

#### Stage 2: Class-Balanced Focal Loss
```python
weights = [w_SPLIT, w_RECT, w_AB]
# Baseado em effective number of samples
CB_FocalLoss(beta=0.9999, gamma=2.0)
```

#### Stage 3-AB: Mixup + Focal Loss
```python
Mixup(alpha=0.4) + FocalLoss(gamma=2.0)
```

### 3. Data Augmentation Strategy

#### Stage 1 (Treino Backbone)
- RandomHorizontalFlip (p=0.5)
- RandomVerticalFlip (p=0.5)
- RandomRotation (90°, 180°, 270°)
- GaussianNoise (σ=0.01)

#### Stage 2 (Macro-classes)
- Todas do Stage 1 +
- Cutout (16x16, p=0.3)
- GridShuffle (4x4, p=0.2)

#### Stage 3-AB (Crítico!)
- **Heavy Augmentation**:
  - HorizontalFlip com label swap (HORZ_A ↔ HORZ_B)
  - VerticalFlip com label swap (VERT_A ↔ VERT_B)
  - Rotation 90° com label rotate (HORZ ↔ VERT)
  - Mixup (α=0.4) entre classes AB
  - CoarseDropout (múltiplos patches)

### 4. Sampling Strategy

#### Stage 1
```python
# Balanceamento 1:1 (NONE vs PARTITION)
WeightedRandomSampler(weights=[0.5, 0.5])
```

#### Stage 2
```python
# Effective Number Sampling
weights = (1 - beta^n_i) / (1 - beta)
# beta=0.9999, favorece classes raras
```

#### Stage 3-AB
```python
# Oversampling 5x para minoritárias
# HORZ_B, VERT_A → repetir 5x
# HORZ_A, VERT_B → repetir 3x
```

### 5. Training Strategy

#### Phase 1: Backbone Pre-training
```
Epochs: 20
LR: 1e-3 → 1e-5 (cosine decay)
Optimizer: AdamW (wd=1e-4)
Task: Stage 1 (Binary)
```

#### Phase 2: Stage 2 Fine-tuning
```
Epochs: 25
LR: 5e-4 → 5e-6 (cosine decay)
Freeze: Backbone (primeiras 2 épocas)
Unfreeze: Backbone (últimas 23 épocas, LR=1e-5)
```

#### Phase 3: Stage 3 Specialists
```
Epochs: 30 (RECT), 50 (AB)
LR: 3e-4 → 3e-6
Freeze: Backbone sempre
Train: Apenas specialist heads

AB Ensemble:
- Treinar 3 modelos com seeds diferentes
- Augmentation aleatório diferente
- Voting na inferência
```

### 6. Threshold Optimization

#### Stage 1 Threshold Search
```python
# Grid search no validation set
thresholds = np.linspace(0.4, 0.7, 31)
for th in thresholds:
    precision, recall, f1 = evaluate(th)
# Escolher threshold que maximiza F1 ou Precision
```

#### Calibração de Probabilidades
```python
# Temperature Scaling (Guo et al., 2017)
# Aprender T no validation set
probs_calibrated = softmax(logits / T)
```

---

## 📊 Dataset Preparation

### Modificações nos Scripts

#### `001_prepare_v6_dataset.py`
```python
# Mudanças vs v5:
1. Stage 2 labels: apenas [SPLIT, RECT, AB] (remove NONE, 1TO4)
2. Oversampling AB classes: 5x minoritárias
3. Validação estratificada: manter distribuição AB
4. Salvar metadados estendidos (confusion matrix baseline)
```

#### `002_prepare_v6_stage3_ab_ensemble.py`
```python
# Novo script específico para AB
1. Criar 3 versões do dataset AB com augmentation diferente
2. Adicionar exemplos sintéticos (Mixup offline)
3. Análise de hard negatives (erros do modelo v5)
```

### Estrutura de Dados

```
pesquisa_v6/
├── v6_dataset/
│   └── block_16/
│       ├── train.pt          # Stage 1+2 combined
│       ├── val.pt
│       └── metadata.json
├── v6_dataset_stage3/
│   ├── RECT/
│   │   └── block_16/
│   │       ├── train.pt
│   │       └── val.pt
│   └── AB/
│       └── block_16/
│           ├── train_v1.pt   # Ensemble member 1
│           ├── train_v2.pt   # Ensemble member 2
│           ├── train_v3.pt   # Ensemble member 3
│           └── val.pt
```

---

## 🧪 Experiments Roadmap

### Fase 1: Baseline Improvements (Semana 1-2)

**Experimento 1.1: Stage 1 Optimization**
- [x] Implementar Focal Loss com γ variável
- [ ] Hard Negative Mining
- [ ] Threshold grid search
- [ ] Temperature Scaling
- **Meta**: Precisão 62%+, F1 68%+

**Experimento 1.2: Stage 2 Redesign**
- [x] Remover NONE da Stage 2 (implementado em data_hub.py)
- [x] Scripts de preparação 001 e 002 (validados)
- [x] Class-Balanced Focal Loss (implementado em losses.py)
- [x] Heavy augmentation (implementado em augmentation.py)
- **Meta**: Macro F1 45%+

### Fase 2: Specialist Improvements (Semana 3-4)

**Experimento 2.1: Stage 3-RECT**
- [ ] Attention mechanism
- [ ] Test-time augmentation (TTA)
- **Meta**: F1 78%+

**Experimento 2.2: Stage 3-AB Ensemble**
- [ ] Implementar 3 modelos independentes
- [ ] Mixup + Heavy Aug
- [ ] Oversampling 5x
- **Meta**: F1 50%+ (dobrar performance v5)

### Fase 3: Advanced Techniques (Semana 5-6)

**Experimento 3.1: Knowledge Distillation**
- [ ] Teacher: ensemble grande
- [ ] Student: modelo compacto
- **Meta**: Manter performance, reduzir latência

**Experimento 3.2: Multi-task Learning**
- [ ] Predição conjunta Stage 2 + Stage 3
- [ ] Shared representations
- **Meta**: Acurácia final 55%+

### Fase 4: Alternative Architectures (Semana 7-8)

**Experimento 4.1: Transformer-based**
- [ ] Vision Transformer (ViT-Tiny)
- [ ] Patch embedding 4x4
- **Meta**: Comparar vs CNN

**Experimento 4.2: Hybrid CNN-Transformer**
- [ ] CNN backbone + Transformer head
- [ ] Self-attention para AB classes
- **Meta**: Best of both worlds

---

## 📁 File Structure

```
pesquisa_v6/
├── PLANO_V6.md                          # Este documento
│
├── v6_pipeline/                         # Core modules
│   ├── __init__.py                      # ✅ 
│   ├── data_hub.py                      # ✅ Dataset utils (standalone)
│   ├── models.py                        # ✅ Backbones + Heads
│   ├── losses.py                        # ✅ Focal, CB, Mixup losses
│   ├── augmentation.py                  # ✅ Aug pipelines
│   ├── ensemble.py                      # ✅ AB ensemble logic
│   └── metrics.py                       # ✅ Evaluation utils
│
├── scripts/                             # Training scripts
│   ├── 001_prepare_v6_dataset.py        # ✅ Validado
│   ├── 002_prepare_v6_stage3_datasets.py # ✅ Validado
│   ├── 003_train_stage1_improved.py     # ✅ Validado
│   ├── 004_train_stage2_redesigned.py   # ✅ Validado
│   ├── 005_train_stage3_rect.py         # ✅ Validado
│   ├── 006_train_stage3_ab_ensemble.py  # ✅ Validado
│   ├── 007_optimize_thresholds.py       # ✅ Validado
│   ├── 008_run_pipeline_eval_v6.py      # ✅ Validado
│   └── 009_compare_v5_v6.py
│
├── experiments/                         # Experiment configs
│   ├── exp1_stage1_optimization.yaml
│   ├── exp2_stage2_redesign.yaml
│   ├── exp3_ab_ensemble.yaml
│   └── exp4_full_pipeline.yaml
│
├── notebooks/                           # Analysis notebooks
│   ├── v6_ablation_study.ipynb
│   ├── v6_error_analysis.ipynb
│   └── v6_vs_v5_comparison.ipynb
│
└── logs/                                # Results
    └── v6_experiments/
        ├── stage1/
        ├── stage2/
        ├── stage3_rect/
        ├── stage3_ab/
        └── pipeline/
```

---

## 🔬 Ablation Studies

### Study 1: Stage 1 Components
| Config | Focal γ | Hard Mining | Threshold | F1 | Precision |
|--------|---------|-------------|-----------|-----|-----------|
| Baseline | 2.0 | No | 0.5 | 65.19% | 53.71% |
| +Focal γ=2.5 | 2.5 | No | 0.5 | ? | ? |
| +Hard Mining | 2.5 | Yes (3:1) | 0.5 | ? | ? |
| +Threshold Opt | 2.5 | Yes | 0.55 | ? | ? |
| Full | 2.5 | Yes | 0.55 | **Target: 70%** | **Target: 65%** |

### Study 2: Stage 2 Redesign
| Config | Classes | Loss | Aug | Macro F1 |
|--------|---------|------|-----|----------|
| v5 Baseline | 5 (NONE,SPLIT,RECT,AB,1TO4) | CE | Basic | 33.41% |
| -NONE, -1TO4 | 3 (SPLIT,RECT,AB) | CE | Basic | ? |
| +CB Focal | 3 | CB-Focal | Basic | ? |
| +Heavy Aug | 3 | CB-Focal | Heavy | ? |
| Full v6 | 3 | CB-Focal | Heavy | **Target: 48%** |

### Study 3: Stage 3-AB Ensemble
| Config | Models | Aug | Sampling | F1 |
|--------|--------|-----|----------|-----|
| v5 Baseline | 1 | Basic | Balanced | 25.26% |
| +Heavy Aug | 1 | Heavy | Balanced | ? |
| +Oversampling | 1 | Heavy | 5x minority | ? |
| +Mixup | 1 | Heavy+Mixup | 5x | ? |
| Ensemble-3 | 3 | Heavy+Mixup | 5x | **Target: 50%** |

---

## 📈 Expected Performance Gains

### Stage-by-Stage Improvements

```
Stage 1:
  v5: F1=65.19%, Prec=53.71%, Rec=82.93%
  v6: F1=70%↑,   Prec=65%↑,    Rec=80%↓
  Estratégia: Trocar recall por precisão (reduzir FP)

Stage 2:
  v5: Macro F1=33.41%
  v6: Macro F1=48%↑ (+44% relativo)
  Estratégia: Simplificar classes, melhorar separabilidade

Stage 3-RECT:
  v5: F1=72.50%
  v6: F1=78%↑ (+8% relativo)
  Estratégia: Atenção + TTA

Stage 3-AB:
  v5: F1=25.26% (CRÍTICO)
  v6: F1=50%↑ (+98% relativo)
  Estratégia: Ensemble + Aug pesado
```

### Final Pipeline Accuracy

```
v5: 39.56%
v6 (conservative): 48% (+21% relativo)
v6 (optimistic): 55% (+39% relativo)
```

---

## 🚀 Implementation Priority

### High Priority (Must Have)
1. ✅ Stage 2 redesign (remover NONE/1TO4)
2. ✅ Stage 3-AB: 2 implementações testadas
   - `006_train_stage3_ab_ensemble_reference.py`: F1=10.35% (arquivado como referência)
   - `006_train_stage3_ab_fgvc.py`: F1=24.50% (aceito, 4/4 classes funcionando)
3. ✅ Focal Loss em todos estágios
4. ✅ Heavy augmentation para AB
5. ⭕ Threshold optimization Stage 1 (próximo: script 007)

### Medium Priority (Should Have)
6. ⭕ Attention mechanism no backbone
7. ⭕ Temperature scaling (calibração)
8. ⭕ Hard negative mining
9. ⭕ Mixup para AB
10. ⭕ TTA (test-time augmentation)

### Low Priority (Nice to Have)
11. ⏸️ Knowledge distillation
12. ⏸️ Transformer heads
13. ⏸️ Multi-task learning
14. ⏸️ Neural architecture search
15. ⏸️ Quantization (deploy)

---

## 📝 Success Criteria

### Critérios Mínimos (Fase 1)
- [ ] Stage 1 F1 ≥ 68%
- [ ] Stage 1 Precision ≥ 62%
- [ ] Stage 2 Macro F1 ≥ 45%
- [ ] Stage 3-AB F1 ≥ 40%
- [ ] Final Accuracy ≥ 48%

### Critérios Ideais (Fase 2)
- [ ] Stage 1 F1 ≥ 72%
- [ ] Stage 1 Precision ≥ 68%
- [ ] Stage 2 Macro F1 ≥ 52%
- [ ] Stage 3-AB F1 ≥ 55%
- [ ] Final Accuracy ≥ 55%

### Critérios Stretch (Fase 3)
- [ ] Stage 1 F1 ≥ 75%
- [ ] Stage 2 Macro F1 ≥ 58%
- [ ] Stage 3-AB F1 ≥ 65%
- [ ] Final Accuracy ≥ 60%

---

## 🔍 Monitoring & Analysis

### Métricas de Acompanhamento
1. **Por Estágio**: Accuracy, F1, Precision, Recall, Confusion Matrix
2. **Por Classe**: F1 individual, Support, Erros específicos
3. **Pipeline**: Acurácia final, Distribuição de erros, Latência
4. **Treino**: Loss curves, Learning rate, Gradients

### Dashboards
```python
# Usar TensorBoard para:
- Loss curves (train/val)
- Confusion matrices (interativo)
- Embedding visualizations (t-SNE)
- Attention maps (heatmaps)

# Usar Weights & Biases para:
- Hyperparameter sweeps
- Model comparisons
- Experiment tracking
```

---

## 📚 References & Inspiration

### Papers
1. Focal Loss (Lin et al., 2017) - Lidar com desbalanceamento
2. Class-Balanced Loss (Cui et al., 2019) - Effective number
3. Mixup (Zhang et al., 2018) - Data augmentation
4. Temperature Scaling (Guo et al., 2017) - Calibração
5. Squeeze-and-Excitation (Hu et al., 2018) - Attention

### Codebases
- timm (PyTorch Image Models) - Backbones
- albumentations - Augmentations
- pytorch-metric-learning - Loss functions

---

## 🤝 Next Steps

1. **Revisar este plano** com o time
2. **Priorizar experimentos** (High → Medium → Low)
3. **Implementar v6_pipeline module** (base code)
4. **Preparar datasets v6** (script 001)
5. **Executar Fase 1** (Experimentos 1.1 e 1.2)
6. **Avaliar resultados** e ajustar plano
7. **Iterar** até atingir critérios de sucesso

---

**Status**: � Em Desenvolvimento (Data prep completo)  
**Última Atualização**: 2025-10-06  
**Responsável**: @chiarorosa  
**Próximo**: Implementar losses.py e models.py
