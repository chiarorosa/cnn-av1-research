# Pipeline v6 - Plano de Desenvolvimento Consolidado

**Data:** 13 de outubro de 2025  
**Status:** 🟡 Em Desenvolvimento - Scripts 8/9 concluídos  
**Última Atualização:** 13/10/2025  
**Responsável:** @chiarorosa

---

## 📋 Sumário Executivo

O pipeline v6 é uma reformulação completa da arquitetura hierárquica, focando em resolver os **3 problemas críticos** identificados na v5:

1. **Stage 1**: Baixa precisão (53.71%) → 27k falsos positivos contaminando o pipeline
2. **Stage 2**: Confusão entre macro-classes (33.41% Macro F1) → Erros propagados
3. **Stage 3-AB**: Colapso total do especialista (25.26% F1) → Classes perdidas

### Progresso Atual (Atualizado 13/10/2025)

| Componente | Status | Resultado |
|------------|--------|-----------|
| **Scripts 001-002** | ✅ Validado | Dataset preparado (152k train / 38k val) |
| **Script 003 (Stage 1)** | ✅ Validado | F1=72.28% (época 19) - **META ATINGIDA** ≥68% |
| **Script 004 (Stage 2)** | ✅ **RESOLVIDO** | F1=46.51% (frozen model época 8) - **META ATINGIDA** ≥45% |
| **Script 005 (Stage 3-RECT)** | ✅ Validado | F1=68.44% (época 12) |
| **Script 006 (Stage 3-AB)** | ✅ Validado | F1=24.50% (4/4 classes, época 6) |
| **Script 007 (Threshold)** | ✅ Validado | threshold=0.45 → F1=72.79% |
| **Script 008 (Pipeline)** | ✅ Validado | Accuracy=47.66% (meta: 48%, gap: -0.34pp) |
| **Script 009 (Compare v5/v6)** | ⏳ Próximo | Último script do pipeline |

---

## 🎯 Objetivos e Metas

### Métricas Alvo (block_16)

| Componente | v5 Atual | v6 Meta Fase 1 | v6 Meta Fase 2 | v6 Obtido (13/10) |
|------------|----------|----------------|----------------|-------------------|
| **Stage 1 F1** | 65.19% | 68-70% | 72-75% | ✅ **72.28%** |
| **Stage 1 Precisão** | 53.71% | 62-65% | 68-72% | ✅ **67.13%** |
| **Stage 2 Macro F1** | 33.41% | 45-50% | 55-60% | ✅ **46.51%** (frozen) |
| **Stage 3-RECT F1** | 72.50% | 75-78% | 80-83% | ⚠️ **68.44%** |
| **Stage 3-AB F1** | 25.26% | 45-50% | 60-65% | ⚠️ **24.50%** |
| **Acurácia Final** | 39.56% | 48-52% | 58-63% | ⚠️ **47.66%** (-0.34pp da meta) |

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
                    │   + SE-Blocks   │
                    │   + Attention   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────────┐
                    │   STAGE 1 (Binary)  │
                    │  NONE vs PARTITION  │
                    │   + Focal Loss      │
                    │   + Threshold=0.45  │
                    │   F1=72.28% ✅      │
                    └──────┬──────────────┘
                           │
                    ┌──────▼──────┐
                    │   IF NONE   │──→ OUTPUT: PARTITION_NONE
                    └─────────────┘
                           │
                    ┌──────▼─────────────┐
                    │   STAGE 2 (3-way)  │
                    │  SPLIT | RECT | AB │
                    │  + CB-Focal Loss   │
                    │  F1=46.51% (frozen)│
                    └──┬────────┬────┬───┘
                       │        │    │
              ┌────────▼──┐  ┌──▼────────────┐
              │  SPLIT    │  │  STAGE 3-RECT │
              │  (direct) │  │  HORZ vs VERT │
              └───────────┘  │  F1=68.44% ✅ │
                             └───────────────┘
                                      │
                             ┌────────▼─────────────┐
                             │   STAGE 3-AB         │
                             │  HORZ_A | HORZ_B |   │
                             │  VERT_A | VERT_B     │
                             │  + FGVC Techniques   │
                             │  F1=24.50% ⚠️        │
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
   - ✅ Threshold otimizado = 0.45 (via script 007)
   - ✅ Calibração de probabilidades (Temperature Scaling)

3. **Stage 3-AB Redesenhado**:
   - ✅ FGVC (Fine-Grained Visual Classification) techniques
   - ✅ Center Loss para compactação intra-classe
   - ✅ CutMix augmentation
   - ✅ Cosine classifier com temperature scaling
   - ✅ 4/4 classes funcionando (vs. v5 com 1 classe colapsada)

---

## 🔴 PROBLEMA CRÍTICO: Stage 2 Catastrophic Forgetting (RESOLVIDO)

### Diagnóstico do Problema

**Treinamento Original (antes ULMFiT):**
- Época 1 (frozen): F1=**47.58%** ✅
- Época 3 (unfrozen): F1=**34-38%** ❌
- Problema: Catastrophic forgetting ao descongelar backbone

**Root Cause: Negative Transfer (Yosinski et al., 2014)**

| Aspecto | Stage 1 | Stage 2 |
|---------|---------|---------|
| **Task** | Binary (NONE vs PARTITION) | 3-way (SPLIT vs RECT vs AB) |
| **Features necessárias** | "Tem partição?" | "Tipo de partição?" |
| **Foco visual** | Detecção de bordas | Padrões geométricos |
| **Distribuição** | 50/50 balanceado | Long-tail (SPLIT minoritária) |
| **Complexidade** | Simples (presença/ausência) | Complexa (geometria) |

**Conclusão:** Tasks são **FUNDAMENTALMENTE DIFERENTES** → Features do Stage 1 prejudicam Stage 2

### Experimentos Realizados

#### ❌ Experimento 1: ULMFiT (07/10/2025)
- **Técnicas:** Gradual unfreezing, discriminative LR, cosine annealing
- **Resultado:** Frozen F1=46.51% → Unfrozen F1=34.12% (-26.6%)
- **Conclusão:** Catastrophic forgetting NÃO foi prevenido

#### ❌ Experimento 2: Train from Scratch (13/10/2025)
- **Implementação:** ImageNet-only pretrained (sem Stage 1 backbone)
- **Resultado:** Best F1=37.38% (época 26)
- **Conclusão:** Elimina catastrophic forgetting, mas F1 inferior ao frozen

#### ❌ Experimento 3: Flatten Architecture (feat/stage2-flatten-9classes)
- **Hipótese H2.1:** Distribution shift causa colapso no pipeline
- **Implementação:** Retreinar Stage 2 com samples filtrados pelo Stage 1 (threshold=0.45)
- **Dataset:** 152k → 3,890 samples (2.55% retenção)
- **Resultado:** F1=6.74% (época 9) - **PIOR que baseline** ❌❌❌
- **Conclusão:** H2.1 **REJEITADA** - Distribution shift NÃO é a causa primária
- **Documentação:** `docs_v6/08_pipeline_aware_training.md` (474 linhas, 9 seções, 16 refs)

### ✅ DECISÃO FINAL: Usar Frozen Model (Época 8)

**Razões Baseadas em Evidências:**

| Abordagem | F1 Obtido | Catastrophic Forgetting | Meta (≥45%) | Status |
|-----------|-----------|------------------------|-------------|--------|
| **ULMFiT Frozen** | **46.51%** | N/A (não unfrozen) | ✅ **SIM** | ⭐ **USADO** |
| ULMFiT Unfrozen | 34.12% | ❌ SIM (-26.6%) | ❌ NÃO | Descartado |
| Train from Scratch | 37.38% | ✅ NÃO (+315%) | ❌ NÃO | Descartado |
| Pipeline-Aware | 6.74% | N/A | ❌ NÃO | ❌ **ABANDONADO** |

**Fundamentação Científica:**
> Raghu et al. (2019) - "Transfusion: Understanding Transfer Learning"  
> "Frozen ImageNet features often outperform fine-tuned models when target task is very different from source task."

**Checkpoint Usado:** `stage2_model_block16_classweights_ep8.pt`

---

## 📊 Performance Pipeline V6 (Script 008 - 13/10/2025)

### Resultados Obtidos

| Métrica | Valor | Status |
|---------|-------|--------|
| **Overall Accuracy** | **47.66%** | ⚠️ -0.34pp da meta (48%) |
| Macro F1 | 13.38% | ⚠️ Muito baixo |
| NONE F1 | 78.88% | ✅ Excelente |
| SPLIT F1 | 8.83% | ⚠️ Baixo |
| **HORZ F1** | **0.00%** | ❌ **Colapsado** |
| VERT F1 | 10.92% | ⚠️ Baixo |
| HORZ_A F1 | 0.00% | ❌ Colapsado |
| HORZ_B F1 | 8.45% | ⚠️ Baixo |
| VERT_A F1 | 0.00% | ❌ Colapsado |
| VERT_B F1 | 0.00% | ❌ Colapsado |

### Problemas Identificados

1. **Erro em Cascata Dominante**
   - Stage 3-RECT: 4.49% accuracy (standalone: 68.44%) → **-93.4% degradação**
   - Stage 3-AB: 1.51% accuracy (standalone: 24.50%) → **-93.8% degradação**
   - Root cause: Stage 2 envia samples errados para Stage 3

2. **HORZ Completamente Colapsado**
   - Ground truth: 9,618 samples (10.59%)
   - Predictions: 0 (0.00%)
   - Hipótese: Stage 3-RECT tem viés extremo para VERT

3. **Classes AB Colapsadas no Pipeline**
   - HORZ_A, VERT_A, VERT_B: 0% predictions
   - Stage 3-AB não funciona em cascata

---

## 🎯 PRÓXIMOS PASSOS: Fechar Gap -0.34pp

### Estratégia: ROI Maximizado
> "Começar com técnicas de baixo custo e alto impacto. Avaliar resultados antes de investir em soluções complexas."

### Fase 1: Quick Wins (3-4 dias) - **RECOMENDADA**

**Objetivo:** Alcançar ≥48% accuracy  
**Ganho Esperado:** +0.3-0.7pp → **Accuracy 47.9-48.4%** ✅

#### 1.1 Investigar Stage 3-RECT Standalone (2h) 🔴 **ALTA PRIORIDADE**

**Problema:**
> "Stage 3-RECT tem F1=68.44% standalone, mas apenas 4.49% no pipeline. Por quê?"

**Hipóteses:**
1. Modelo tem viés extremo para VERT (explica HORZ=0%)
2. Modelo não generaliza para samples enviados erroneamente por Stage 2
3. Dataset de treinamento desbalanceado

**Protocolo:**
```bash
# Script: pesquisa_v6/scripts/009_diagnose_stage3_rect.py
python3 009_diagnose_stage3_rect.py \
  --model pesquisa_v6/logs/v6_experiments/stage3_rect/stage3_rect_model_best.pt \
  --dataset pesquisa_v6/v6_dataset_stage3/RECT/block_16/val.pt

# Analisar:
# - Confusion matrix standalone
# - F1 per-class (HORZ vs VERT)
# - Class distribution no dataset train
```

**Ações baseadas em resultados:**
- Se HORZ F1 < 50% standalone → Retreinar com weighted loss (1 dia)
- Se HORZ samples < 40% train → Rebalancear dataset (usar sampler)

**Ganho:** +0.1-0.3pp

#### 1.2 Threshold Grid Search (2h) 🔴 **ALTA PRIORIDADE**

**Problema:**
> "Threshold Stage 1 = 0.45 foi otimizado isoladamente. No pipeline, pode estar enviando muitos false positives para Stage 2."

**Protocolo:**
```python
# Script: pesquisa_v6/scripts/010_threshold_grid_search.py
thresholds_stage1 = [0.40, 0.45, 0.50, 0.55]
results = []

for th1 in thresholds_stage1:
    accuracy = run_pipeline(stage1_threshold=th1)
    results.append({'th1': th1, 'accuracy': accuracy})

best = max(results, key=lambda x: x['accuracy'])
```

**Custo:** 40 min runs + 1h análise = 2h  
**Ganho:** +0.1-0.3pp

#### 1.3 Stage 2 Strong Data Augmentation (1 dia) 🟡 **MÉDIA PRIORIDADE**

**Problema:**
> "Stage 2 F1=46.51% pode melhorar com augmentation mais agressiva."

**Técnicas:**
- **MixUp** (Zhang et al., 2018): α=0.4
- **CutMix** (Yun et al., 2019): β=1.0
- **Geometric Augmentations**: Flips + Rotation

**Protocolo:**
```bash
python3 004_train_stage2_redesigned.py \
  --epochs 30 \
  --mixup-alpha 0.4 \
  --cutmix-beta 1.0 \
  --output-dir stage2_scratch_augstrong
```

**Custo:** 6h treinamento + 2h análise = 1 dia  
**Ganho:** +0.2-0.4pp

**Ganho Total Fase 1:** +0.4-1.0pp → **Accuracy 48.0-48.7%** ✅

---

## 🔧 Melhorias Técnicas Implementadas (v6)

### 1. Backbone Upgrade
```python
# ResNet-18 + SE-Blocks + Dropout progressivo
- Squeeze-and-Excitation blocks (channel attention)
- Spatial Attention Module (SAM)
- Dropout progressivo: [0.1, 0.2, 0.3, 0.4]
- Group Normalization
```

### 2. Loss Functions

- **Stage 1:** FocalLoss(α=0.25, γ=2.5) + HardNegativeMining(ratio=3:1)
- **Stage 2:** ClassBalancedFocalLoss(β=0.9999, γ=2.0)
- **Stage 3-AB:** Mixup(α=0.4) + FocalLoss(γ=2.0)

### 3. Data Augmentation Strategy

| Stage | Augmentation |
|-------|--------------|
| Stage 1 | Basic: HFlip, VFlip, Rotation, GaussianNoise |
| Stage 2 | Medium: Stage 1 + Cutout, GridShuffle |
| Stage 3-AB | **Heavy**: Flips com label swap, Mixup, CoarseDropout |

### 4. Training Strategy

#### Stage 1: Backbone Pre-training
- Epochs: 20, LR: 1e-3 → 1e-5 (cosine decay)
- Result: F1=72.28% ✅

#### Stage 2: Fine-tuning (FROZEN-ONLY)
- Epochs: 8 (frozen), LR: 5e-4
- Result: F1=46.51% ✅
- **Decisão:** NÃO descongelar (negative transfer)

#### Stage 3: Specialists
- Epochs: 30 (RECT), 50 (AB)
- LR: 3e-4 → 3e-6
- Freeze: Backbone sempre (evita catastrophic forgetting)

---

## 📁 File Structure

```
pesquisa_v6/
├── PLANO_v6_Out.md                      # ⭐ Este documento (consolidado)
│
├── v6_pipeline/                         # Core modules
│   ├── __init__.py                      # ✅ 
│   ├── data_hub.py                      # ✅ Dataset utils
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
│   ├── 004_train_stage2_redesigned.py   # ✅ Validado (frozen model)
│   ├── 005_train_stage3_rect.py         # ✅ Validado
│   ├── 006_train_stage3_ab_fgvc.py      # ✅ Validado
│   ├── 007_optimize_thresholds.py       # ✅ Validado
│   ├── 008_run_pipeline_eval_v6.py      # ✅ Validado
│   └── 009_compare_v5_v6.py             # ⏳ Próximo
│
├── docs_v6/                             # Documentação técnico-científica
│   ├── 04_experimento_train_from_scratch.md  # Train from scratch (F1=37.38%)
│   ├── 06_arquitetura_flatten_9classes.md    # Flatten architecture design
│   ├── 07_flatten_pipeline_evaluation.md     # Baseline flatten (F1=10.24%)
│   └── 08_pipeline_aware_training.md         # Pipeline-aware NEGATIVO (F1=6.74%)
│
└── logs/                                # Results
    └── v6_experiments/
        ├── stage1/                      # Stage 1 checkpoints
        ├── stage2/                      # Stage 2 frozen model (USADO)
        ├── stage2_scratch/              # Stage 2 train from scratch
        ├── stage2_pipeline_aware/       # Pipeline-aware (ABANDONADO)
        ├── stage3_rect/                 # Stage 3-RECT checkpoints
        └── stage3_ab/                   # Stage 3-AB FGVC checkpoints
```

---

## 🧪 Experimentos Flatten (ARQUIVADOS - NEGATIVOS)

**Branch:** `feat/stage2-flatten-9classes` (merged to main em 904e0aa)  
**Duração:** ~3 dias (10-13 outubro 2025)  
**Status:** ❌ **ABANDONADO** - Resultados negativos

### Resumo dos Experimentos

| Experimento | F1-macro | Accuracy | Status |
|-------------|----------|----------|--------|
| Stage 2 isolado (004b) | 31.65% | 37.17% | ✅ Funciona |
| Pipeline baseline (008b) | 10.24% | 57.93% | ⚠️ Acc boa, F1 ruim |
| Pipeline-aware (004c) | 6.74% | 8.50% | ❌❌❌ **PIOR** |

### Hipótese H2.1 (distribution shift): **REJEITADA**

**Hipótese:**
> "Stage 2 colapsa no pipeline porque treina com distribuição balanceada mas recebe distribuição filtrada pelo Stage 1. Solução: retreinar Stage 2 com samples filtrados."

**Resultado:**
- F1 piorou de 31.65% → 6.74% (-78.7% degradação)
- Hipótese H2.1 **falsificada experimentalmente**

**Root Causes Identificados:**
1. **Dataset too small:** 3,890 samples / 11.3M params = 1:2,900 ratio (need 1:10-100)
2. **Negative transfer:** Stage 1 binary features incompatíveis com Stage 2 multi-class
3. **Architectural flaw:** Objetivos conflitantes (Stage 1 vs Stage 2)

**Conclusão Científica:**
> Flatten architecture fundamentalmente falha para predição de partições AV1. Negative transfer confirmado experimentalmente. Recommendation: Retornar à hierárquica V6.

**Documentação Completa:** `docs_v6/08_pipeline_aware_training.md` (474 linhas, 9 seções, 16 referências)

---

## 📝 Success Criteria

### ✅ Critérios Mínimos (Fase 1) - **QUASE ATINGIDOS**

- [x] Stage 1 F1 ≥ 68% → **72.28%** ✅
- [x] Stage 1 Precision ≥ 62% → **67.13%** ✅
- [x] Stage 2 Macro F1 ≥ 45% → **46.51%** ✅
- [ ] Stage 3-AB F1 ≥ 40% → **24.50%** ❌ (gap: -15.5pp)
- [ ] Final Accuracy ≥ 48% → **47.66%** ⚠️ (gap: -0.34pp)

### ⭕ Critérios Ideais (Fase 2)

- [ ] Stage 1 F1 ≥ 72% → **72.28%** ✅
- [ ] Stage 1 Precision ≥ 68% → **67.13%** ⚠️ (gap: -0.87pp)
- [ ] Stage 2 Macro F1 ≥ 52% → **46.51%** ❌ (gap: -5.5pp)
- [ ] Stage 3-AB F1 ≥ 55% → **24.50%** ❌ (gap: -30.5pp)
- [ ] Final Accuracy ≥ 55% → **47.66%** ❌ (gap: -7.34pp)

---

## 📚 Referências Científicas

### Negative Transfer & Transfer Learning
1. **Yosinski et al., 2014:** "How transferable are features in deep neural networks?"
2. **Raghu et al., 2019:** "Transfusion: Understanding Transfer Learning"
3. **Kornblith et al., 2019:** "Do Better ImageNet Models Transfer Better?"

### Loss Functions
4. **Lin et al., 2017:** "Focal Loss for Dense Object Detection"
5. **Cui et al., 2019:** "Class-Balanced Loss Based on Effective Number of Samples"
6. **Zhang et al., 2018:** "mixup: Beyond Empirical Risk Minimization"

### Catastrophic Forgetting
7. **Goodfellow et al., 2013:** "An Empirical Investigation of Catastrophic Forgetting"
8. **Howard & Ruder, 2018:** "Universal Language Model Fine-tuning (ULMFiT)"
9. **Kirkpatrick et al., 2017:** "Overcoming catastrophic forgetting"

### Data Augmentation & Robustness
10. **Yun et al., 2019:** "CutMix: Regularization Strategy to Train Strong Classifiers"
11. **DeVries & Taylor, 2017:** "Improved Regularization with Cutout"
12. **Hendrycks et al., 2019:** "Using Pre-Training Can Improve Model Robustness"

### Attention Mechanisms
13. **Hu et al., 2018:** "Squeeze-and-Excitation Networks"
14. **Woo et al., 2018:** "CBAM: Convolutional Block Attention Module"

### Model Calibration
15. **Guo et al., 2017:** "On Calibration of Modern Neural Networks"
16. **Müller et al., 2019:** "When Does Label Smoothing Help?"

---

## 🤝 Próximo Passo Imediato

### Decisão: Executar Fase 1 (3-4 dias)

**Sequência Recomendada:**

```bash
# 1. Dia 1 manhã: Diagnose Stage 3-RECT (2h)
python3 pesquisa_v6/scripts/009_diagnose_stage3_rect.py

# 2. Dia 1 tarde: Threshold grid search (2h)
python3 pesquisa_v6/scripts/010_threshold_grid_search.py

# 3. Dia 2-3: Strong augmentation (1 dia) - SE NECESSÁRIO
python3 pesquisa_v6/scripts/004_train_stage2_redesigned.py \
  --mixup-alpha 0.4 --cutmix-beta 1.0 --epochs 30

# 4. Dia 4: Pipeline re-evaluation + análise
python3 pesquisa_v6/scripts/008_run_pipeline_eval_v6.py
```

**Checkpoint de Decisão:**
- ✅ Se accuracy ≥ 48%: **PARAR**, documentar, finalizar
- ⚠️ Se 47.8-48%: Considerar 1.3 (augmentation)
- ❌ Se < 47.8%: Avaliar Fase 2 ou aceitar resultado

**Meta:** Fechar gap de -0.34pp e atingir ≥48% accuracy para finalizar Pipeline v6.

---

**Status Final:** 🟡 Em Desenvolvimento - 8/9 scripts validados  
**Próximo:** Script 009 (compare v5/v6) OU Fase 1 Quick Wins  
**Última Atualização:** 13 de outubro de 2025  
**Responsável:** @chiarorosa
