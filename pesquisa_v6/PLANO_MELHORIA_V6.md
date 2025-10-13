# Plano de Melhoria V6 - Pipeline Accuracy 47.66% → 48%+

**Data:** 13 de outubro de 2025  
**Status Atual:** Pipeline V6 com Train from Scratch - Accuracy 47.66% (meta: 48%, gap: -0.34pp)  
**Objetivo:** Fechar gap de -0.34pp com técnicas viáveis e eficientes

---

## 📊 Situação Atual

### Performance Pipeline V6

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
   - Hipótese: Stage 2 → RECT, Stage 3-RECT sempre prediz VERT?

3. **Stage 2 F1 Ainda Baixo**
   - Train from Scratch: F1=37.38%
   - ~62% de erros propagam para Stage 3

4. **Classes AB Colapsadas**
   - HORZ_A, VERT_A, VERT_B: 0% predictions
   - Stage 3-AB não funciona no pipeline

---

## 🎯 Estratégia: Abordagem Faseada

### Princípio: ROI Maximizado
> "Começar com técnicas de baixo custo e alto impacto. Avaliar resultados antes de investir em soluções complexas."

### Meta de Cada Fase
- **Fase 1:** Alcançar ≥48% (fechar gap atual)
- **Fase 2:** Alcançar 48.5-49% (se Fase 1 parcialmente bem-sucedida)
- **Fase 3:** Alcançar 49-50% (otimização avançada)

---

## 📋 FASE 1: Quick Wins (3-4 dias)

**Objetivo:** Fechar gap de -0.34pp com técnicas simples  
**Custo Total:** 3-4 dias  
**Ganho Esperado:** +0.3-0.7pp → **Accuracy 47.9-48.4%** ✅

---

### 1.1 Investigar Stage 3-RECT Standalone (2h)

**Problema a Resolver:**
> "Stage 3-RECT tem F1=68.44% standalone, mas apenas 4.49% no pipeline. Por quê?"

**Hipóteses:**
1. Modelo tem viés extremo para VERT (explica HORZ=0%)
2. Modelo não generaliza para samples que Stage 2 envia erroneamente
3. Dataset de treinamento Stage 3-RECT desbalanceado

**Protocolo Experimental:**

```bash
# Script: pesquisa_v6/scripts/009_diagnose_stage3_rect.py

# 1. Avaliar Stage 3-RECT isoladamente no validation set
python3 009_diagnose_stage3_rect.py \
  --model pesquisa_v6/logs/v6_experiments/stage3_rect/stage3_rect_model_best.pt \
  --dataset pesquisa_v6/v6_dataset_stage3/RECT/block_16/val.pt

# 2. Analisar:
#    - Confusion matrix standalone
#    - Per-class F1 (HORZ vs VERT)
#    - Class distribution no dataset de treino
```

**Análises a Fazer:**
- [ ] Confusion matrix: HORZ predito como VERT?
- [ ] F1 per-class: HORZ F1 < 50%?
- [ ] Dataset train: HORZ samples < 40%?

**Ações Baseadas em Resultados:**

| Resultado | Ação |
|-----------|------|
| **HORZ F1 < 50% standalone** | Retreinar Stage 3-RECT com weighted loss (1 dia) |
| **HORZ samples < 40% train** | Rebalancear dataset Stage 3-RECT (usar sampler) |
| **HORZ F1 ≥ 60% standalone** | Problema é Stage 2 routing (não Stage 3 bias) |

**Ganho Esperado:** +0.1-0.3pp se retreinar com fix

**Literatura:**
- Johnson & Khoshgoftaar, 2019: Survey on Deep Learning with Class Imbalance
- Buda et al., 2018: Weighted Loss Functions for Class Imbalance

---

### 1.2 Threshold Grid Search (2h)

**Problema a Resolver:**
> "Threshold Stage 1 = 0.45 foi otimizado isoladamente. No pipeline, pode estar enviando muitos false positives para Stage 2."

**Hipótese:**
- Threshold mais alto (0.50) → menos false positives Stage 1 → Stage 2 recebe samples melhores
- Threshold mais baixo (0.40) → mais recall Stage 1 → pode ajudar classes minoritárias

**Protocolo Experimental:**

```python
# Script: pesquisa_v6/scripts/010_threshold_grid_search.py

thresholds_stage1 = [0.40, 0.45, 0.50, 0.55]
results = []

for th1 in thresholds_stage1:
    accuracy = run_pipeline(stage1_threshold=th1)
    results.append({'th1': th1, 'accuracy': accuracy})

# Encontrar melhor threshold
best = max(results, key=lambda x: x['accuracy'])
print(f"Best threshold: {best['th1']}, Accuracy: {best['accuracy']:.2%}")
```

**Custo Computacional:**
- 4 thresholds × 10 min/run = 40 min
- + 1h análise detalhada
- **Total:** 2h

**Ganho Esperado:** +0.1-0.3pp

**Literatura:**
- Davis & Goadrich, 2006: The Relationship Between Precision-Recall and ROC Curves
- Saito & Rehmsmeier, 2015: Precision-Recall Plot for Imbalanced Datasets

---

### 1.3 Stage 2 Strong Data Augmentation (1 dia)

**Problema a Resolver:**
> "Stage 2 F1=37.38% muito baixo. Augmentation agressiva pode melhorar generalização para HORZ vs VERT geometry."

**Técnicas a Aplicar:**

#### MixUp (Zhang et al., 2018)
```python
# Interpolar entre 2 samples
lambda_param = np.random.beta(0.4, 0.4)  # α=0.4
x_mixed = lambda_param * x1 + (1 - lambda_param) * x2
y_mixed = lambda_param * y1 + (1 - lambda_param) * y2
```

**Benefit:** Força modelo a aprender features mais robustas (edges, geometry)

#### CutMix (Yun et al., 2019)
```python
# Cortar região de x1 e colar em x2
lam = np.random.beta(1.0, 1.0)  # β=1.0
bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
x[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2]
y = lam * y1 + (1 - lam) * y2
```

**Benefit:** Aprende a focar em regiões locais (partições assimétricas)

#### Geometric Augmentations
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # HORZ vs VERT geometry
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-10, 10)),
])
```

**Benefit:** Invariância a orientação (ajuda RECT vs AB)

**Protocolo Experimental:**

```bash
# Script: pesquisa_v6/scripts/004_train_stage2_redesigned.py (modificado)

python3 004_train_stage2_redesigned.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --epochs 30 \
  --batch-size 128 \
  --mixup-alpha 0.4 \
  --cutmix-beta 1.0 \
  --mixup-prob 0.3 \
  --cutmix-prob 0.5 \
  --geometric-aug \
  --output-dir pesquisa_v6/logs/v6_experiments/stage2_scratch_augstrong
```

**Hyperparameters:**
- MixUp: α=0.4, prob=0.3
- CutMix: β=1.0, prob=0.5
- Geometric: flip + rotation
- Epochs: 30
- LR: 1e-4 (backbone), 3e-4 (head)

**Custo:**
- 6h treinamento (30 epochs)
- 1h análise
- 1h pipeline re-evaluation
- **Total:** ~1 dia

**Ganho Esperado:**
- Stage 2 F1: 37% → 40-42% (+8-13%)
- Pipeline accuracy: +0.2-0.4pp

**Literatura:**
- Zhang et al., 2018: mixup - Beyond Empirical Risk Minimization
- Yun et al., 2019: CutMix - Regularization Strategy to Train Strong Classifiers
- DeVries & Taylor, 2017: Improved Regularization of CNNs with Cutout

---

### Resumo Fase 1

| Técnica | Custo | Ganho Esperado | Prioridade |
|---------|-------|----------------|------------|
| **1.1 Diagnose Stage 3-RECT** | 2h | +0.1-0.3pp | 🔴 Alta |
| **1.2 Threshold Grid Search** | 2h | +0.1-0.3pp | 🔴 Alta |
| **1.3 Stage 2 Strong Aug** | 1 dia | +0.2-0.4pp | 🟡 Média |

**Ganho Total Fase 1:** +0.4-1.0pp → **Accuracy 48.0-48.7%** ✅

**Decisão Fase 1:**
- Se accuracy ≥ 48%: **PARAR** e documentar
- Se 47.8-48%: **Tentar** 1.3 (augmentation)
- Se < 47.8%: **Avançar** para Fase 2

---

## 📋 FASE 2: Medium-Cost Optimizations (5-7 dias)

**Condição de Entrada:** Fase 1 < 48%  
**Objetivo:** Alcançar 48.5-49%  
**Custo Total:** 5-7 dias  
**Ganho Esperado:** +0.5-1.0pp

---

### 2.1 Noise Injection em Stage 3 (3 dias)

**Problema a Resolver:**
> "Stage 3 treinou apenas com samples 'limpos'. No pipeline, recebe samples errados do Stage 2 e colapsa."

**Hipótese:**
- Treinar Stage 3 com 20% "dirty samples" → aprende robustez
- Simula distribuição real que Stage 3 receberá no pipeline

**Técnica: Adversarial Data Augmentation**

#### Stage 3-RECT (HORZ vs VERT)
```python
# Durante treinamento
for epoch in range(epochs):
    for batch in dataloader_RECT:
        x_rect, y_rect = batch
        
        # 20% das vezes, injetar sample AB
        if np.random.rand() < 0.2:
            idx = np.random.randint(len(dataset_AB))
            x_noise, y_noise = dataset_AB[idx]
            
            # Substituir um sample RECT por AB
            x_rect[0] = x_noise
            y_rect[0] = np.random.choice([0, 1])  # Random label
            
        loss = criterion(model(x_rect), y_rect)
```

**Resultado Esperado:**
- Stage 3-RECT aprende: "Se recebo AB, não confie na predição"
- Pode outputar probabilidades mais uniformes para samples OOD

#### Stage 3-AB (4-way)
```python
# Injetar 20% RECT samples durante treinamento
if np.random.rand() < 0.2:
    x_noise, _ = dataset_RECT[random_idx]
    x_ab[0] = x_noise
    y_ab[0] = np.random.choice([0, 1, 2, 3])  # Random AB class
```

**Protocolo Experimental:**

```bash
# 1. Retreinar Stage 3-RECT com noise
python3 005_train_stage3_rect.py \
  --noise-injection 0.2 \
  --noise-source AB \
  --epochs 25 \
  --output-dir stage3_rect_robust

# 2. Retreinar Stage 3-AB com noise
python3 006_train_stage3_ab_fgvc.py \
  --noise-injection 0.2 \
  --noise-source RECT \
  --epochs 25 \
  --output-dir stage3_ab_robust

# 3. Avaliar pipeline com modelos robustos
python3 008_run_pipeline_eval_v6.py \
  --stage3-rect-model stage3_rect_robust/model_best.pt \
  --stage3-ab-models stage3_ab_robust/model_best.pt ...
```

**Custo:**
- 1 dia retreino Stage 3-RECT (25 epochs)
- 1 dia retreino Stage 3-AB (25 epochs)
- 0.5 dia pipeline evaluation
- 0.5 dia análise
- **Total:** 3 dias

**Ganho Esperado:**
- Stage 3-RECT pipeline accuracy: 4.49% → 6-8% (+33-78%)
- Stage 3-AB pipeline accuracy: 1.51% → 2-3% (+32-98%)
- Overall pipeline: +0.3-0.6pp

**Literatura:**
- Hendrycks et al., 2019: Using Pre-Training Can Improve Model Robustness and Uncertainty
- Geirhos et al., 2018: Generalisation in Humans and Deep Neural Networks
- Recht et al., 2019: Do ImageNet Classifiers Generalize to ImageNet?

**Limitação:**
- Pode degradar performance standalone
- Trade-off: robustez vs accuracy em distribuição limpa

---

### 2.2 Focal Loss Hyperparameter Tuning (2 dias)

**Problema a Resolver:**
> "CB-Focal Loss atual usa γ=2.0 (default). RECT vs AB são hard examples. γ maior pode ajudar."

**Técnica: Grid Search Focal Loss Parameters**

#### Parâmetros a Otimizar

1. **γ (gamma):** Controla foco em hard examples
   - Atual: 2.0
   - Testar: [1.5, 2.0, 2.5, 3.0]
   - Maior γ → mais peso em hard examples (RECT vs AB confusion)

2. **α (alpha):** Balanceamento de classes
   - Atual: 0.25
   - Testar: [0.15, 0.25, 0.35]
   - Ajustar para compensar imbalance SPLIT (15.7%) vs RECT (25.9%)

**Protocolo Experimental:**

```python
# Script: pesquisa_v6/scripts/011_focal_loss_tuning.py

gammas = [1.5, 2.0, 2.5, 3.0]
alphas = [0.15, 0.25, 0.35]

results = []
for gamma in gammas:
    for alpha in alphas:
        # Treinar Stage 2 com novos params
        model = train_stage2(gamma=gamma, alpha=alpha, epochs=30)
        f1 = evaluate(model)
        results.append({'gamma': gamma, 'alpha': alpha, 'f1': f1})

# Encontrar melhor combinação
best = max(results, key=lambda x: x['f1'])
```

**Custo:**
- 12 combinações × 3h treinamento = 36h (~1.5 dias)
- + 0.5 dia análise
- **Total:** 2 dias

**Ganho Esperado:**
- Stage 2 F1: +2-5pp
- Pipeline accuracy: +0.2-0.4pp

**Literatura:**
- Lin et al., 2017: Focal Loss for Dense Object Detection
- Cui et al., 2019: Class-Balanced Loss Based on Effective Number of Samples
- Mukhoti et al., 2020: Calibrating Deep Neural Networks using Focal Loss

---

### 2.3 Ensemble Stage 2 (Frozen + Scratch) (7 dias)

**Problema a Resolver:**
> "Frozen melhor para AB (7.78%), Scratch melhor para RECT (4.49%). Por que não usar ambos?"

**Técnica: Confidence-Based Ensemble**

#### Estratégia 1: Routing por Confidence

```python
def ensemble_stage2(x):
    # Executar ambos modelos
    logits_frozen = model_frozen(x)
    logits_scratch = model_scratch(x)
    
    probs_frozen = softmax(logits_frozen)
    probs_scratch = softmax(logits_scratch)
    
    pred_frozen = argmax(probs_frozen)
    pred_scratch = argmax(probs_scratch)
    
    conf_frozen = max(probs_frozen)
    conf_scratch = max(probs_scratch)
    
    # Regra 1: Se ambos concordam, use qualquer um
    if pred_frozen == pred_scratch:
        return pred_frozen
    
    # Regra 2: Se discordam, use especialização
    if pred_frozen == AB and conf_frozen > 0.6:
        return AB  # Frozen é especialista em AB
    elif pred_scratch == RECT and conf_scratch > 0.5:
        return RECT  # Scratch é especialista em RECT
    else:
        # Use maior confidence
        return pred_frozen if conf_frozen > conf_scratch else pred_scratch
```

#### Estratégia 2: Weighted Averaging

```python
def ensemble_stage2_weighted(x):
    logits_frozen = model_frozen(x)
    logits_scratch = model_scratch(x)
    
    # Pesos aprendidos por validação
    w_frozen = {'AB': 0.7, 'RECT': 0.3, 'SPLIT': 0.5}
    w_scratch = {'AB': 0.3, 'RECT': 0.7, 'SPLIT': 0.5}
    
    # Weighted average por classe
    logits_ensemble = w_frozen * logits_frozen + w_scratch * logits_scratch
    return argmax(softmax(logits_ensemble))
```

**Protocolo Experimental:**

```bash
# 1. Implementar ensemble no pipeline
# Modificar 008_run_pipeline_eval_v6.py

# 2. Tunar thresholds de confidence
python3 012_tune_ensemble_thresholds.py \
  --frozen-model stage2/stage2_model_best.pt \
  --scratch-model stage2_scratch/stage2_model_best.pt \
  --search-space "conf_frozen_ab:[0.5,0.7],conf_scratch_rect:[0.4,0.6]"

# 3. Avaliar pipeline com ensemble
python3 008_run_pipeline_eval_v6.py \
  --stage2-ensemble \
  --stage2-frozen stage2/stage2_model_best.pt \
  --stage2-scratch stage2_scratch/stage2_model_best.pt \
  --ensemble-config config.json
```

**Custo:**
- 2 dias implementação ensemble
- 2 dias tuning thresholds (grid search)
- 2 dias validação e análise
- 1 dia debugging
- **Total:** 7 dias

**Ganho Esperado:**
- Combina best of both: AB de frozen + RECT de scratch
- Pipeline accuracy: +0.5-1.0pp

**Literatura:**
- Dietterich, 2000: Ensemble Methods in Machine Learning
- Zhou, 2012: Ensemble Methods - Foundations and Algorithms
- Wolpert, 1992: Stacked Generalization

**Limitação:**
- Dobra inference time (2 modelos)
- Complexidade de manutenção

---

### Resumo Fase 2

| Técnica | Custo | Ganho Esperado | Prioridade |
|---------|-------|----------------|------------|
| **2.1 Noise Injection Stage 3** | 3 dias | +0.3-0.6pp | 🔴 Alta |
| **2.2 Focal Loss Tuning** | 2 dias | +0.2-0.4pp | 🟡 Média |
| **2.3 Ensemble Frozen+Scratch** | 7 dias | +0.5-1.0pp | 🟢 Baixa |

**Ganho Total Fase 2:** +0.5-1.0pp → **Accuracy 48.4-49.4%**

**Decisão Fase 2:**
- Se accuracy ≥ 48.5%: **PARAR** e documentar
- Se < 48.5%: **Avaliar** custo-benefício Fase 3

---

## 📋 FASE 3: Advanced Optimizations (1-2 semanas)

**Condição de Entrada:** Fase 2 < 48.5% E vontade de investir 1-2 semanas  
**Objetivo:** Alcançar 49-50%  
**Custo Total:** 1-2 semanas  
**Ganho Esperado:** +0.5-1.5pp

---

### 3.1 Multi-Task Learning Stage 2 (10 dias)

**Problema a Resolver:**
> "Stage 2 classifica SPLIT vs RECT vs AB, mas não aprende geometria interna de RECT (HORZ vs VERT). Isso causa má separação."

**Técnica: Dual-Head Architecture**

```python
class MultiTaskStage2(nn.Module):
    def __init__(self):
        self.backbone = ImprovedBackbone()  # ResNet-18 + SE + Attention
        
        # Head principal: 3-way (SPLIT, RECT, AB)
        self.head_3way = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 3)
        )
        
        # Head auxiliar: 2-way (HORZ, VERT) - apenas para samples RECT
        self.head_rect_geometry = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        pred_3way = self.head_3way(features)
        pred_rect = self.head_rect_geometry(features)
        return pred_3way, pred_rect
```

**Loss Function:**
```python
def multitask_loss(pred_3way, pred_rect, y_3way, y_rect, is_rect_mask):
    # Loss principal (todos samples)
    loss_3way = CB_FocalLoss(pred_3way, y_3way)
    
    # Loss auxiliar (apenas samples RECT)
    loss_rect = nn.CrossEntropyLoss()(
        pred_rect[is_rect_mask],
        y_rect[is_rect_mask]
    )
    
    # Combinar com pesos
    return loss_3way + 0.5 * loss_rect
```

**Vantagens:**
1. Backbone aprende features úteis para HORZ vs VERT
2. Melhora separação RECT vs AB (geometria compartilhada)
3. Regularização implícita (multi-task learning)

**Protocolo Experimental:**

```bash
# 1. Preparar dataset com labels auxiliares
python3 013_prepare_multitask_dataset.py \
  --input v6_dataset/block_16 \
  --output v6_dataset_multitask/block_16

# 2. Treinar modelo multi-task
python3 014_train_stage2_multitask.py \
  --dataset-dir v6_dataset_multitask/block_16 \
  --epochs 40 \
  --loss-weight-rect 0.5 \
  --output-dir stage2_multitask

# 3. Avaliar pipeline
python3 008_run_pipeline_eval_v6.py \
  --stage2-model stage2_multitask/model_best.pt
```

**Custo:**
- 3 dias implementação (dataset prep + model + training loop)
- 5 dias treinamento + tuning (40-50 epochs, grid search loss weights)
- 2 dias validação e análise
- **Total:** 10 dias

**Ganho Esperado:**
- Stage 2 F1: 37% → 42-45% (+13-21%)
- Melhor separação RECT → menos erro cascata
- Pipeline accuracy: +0.5-1.0pp

**Literatura:**
- Caruana, 1997: Multitask Learning
- Ruder, 2017: An Overview of Multi-Task Learning in Deep Neural Networks
- Kendall et al., 2018: Multi-Task Learning Using Uncertainty to Weigh Losses

**Risco:**
- Task interference (auxiliary task degrada primary task)
- Precisa tunar loss weight cuidadosamente

---

### 3.2 Vision Transformer Stage 2 (14 dias)

**Problema a Resolver:**
> "ResNet-18 é CNN pura (receptive field limitado). ViT com self-attention pode capturar melhor padrões globais de partição."

**Técnica: Replace CNN Backbone with Transformer**

```python
from transformers import ViTModel, ViTConfig

class ViTStage2(nn.Module):
    def __init__(self):
        # ViT-Small configuration
        config = ViTConfig(
            image_size=16,
            patch_size=2,  # 16/2 = 8 patches
            num_channels=1,  # Grayscale
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
        )
        
        self.vit = ViTModel(config)
        self.head = nn.Linear(384, 3)
        
    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        return self.head(cls_token)
```

**Vantagens:**
1. Self-attention captura relações espaciais long-range
2. Melhor para padrões geométricos (HORZ, VERT, asymmetric)
3. State-of-the-art em muitas tarefas

**Desafios:**
1. **Data-hungry:** ViT precisa de muito mais dados que CNN
   - Nosso dataset: ~130k samples
   - ViT typical: >1M samples
2. **Convergência lenta:** 50-100 epochs
3. **Hyperparameter tuning complexo**

**Protocolo Experimental:**

```bash
# 1. Implementar ViT Stage 2
# (4 dias desenvolvimento)

# 2. Treinar com técnicas de data efficiency
python3 015_train_stage2_vit.py \
  --epochs 100 \
  --batch-size 64 \
  --mixup-alpha 0.5 \
  --cutmix-beta 1.0 \
  --label-smoothing 0.1 \
  --warmup-epochs 10 \
  --output-dir stage2_vit

# 3. Avaliar
python3 008_run_pipeline_eval_v6.py \
  --stage2-model stage2_vit/model_best.pt
```

**Custo:**
- 4 dias implementação (ViT model + data pipeline)
- 6 dias treinamento (100 epochs, slow convergence)
- 2 dias hyperparameter tuning
- 2 dias análise
- **Total:** 14 dias

**Ganho Esperado:**
- **Best case:** Stage 2 F1 37% → 45%+ (+21%)
- **Worst case:** Underfitting (não converge com poucos dados)
- Pipeline accuracy: +0.5-1.5pp (se funcionar)

**Literatura:**
- Dosovitskiy et al., 2021: An Image is Worth 16×16 Words - ViT
- Liu et al., 2021: Swin Transformer - Hierarchical Vision Transformer
- Touvron et al., 2021: Training Data-Efficient Image Transformers

**Recomendação:**
⚠️ **Risco alto** - ViT pode não funcionar com 130k samples
✅ **Alternativa:** Testar Swin Transformer (mais data-efficient)

---

### 3.3 Stage 2.5 - Intermediate Refinement (7 dias)

**Problema a Resolver:**
> "Stage 3-RECT recebe samples ruins de Stage 2 e colapsa. E se inserirmos um stage intermediário robusto?"

**Técnica: Add Intermediate Stage for RECT**

**Nova Arquitetura:**
```
Stage 1: NONE vs PARTITION
  ↓
Stage 2: SPLIT vs RECT vs AB
  ↓
  ├─ SPLIT → Output SPLIT
  ├─ AB → Stage 3-AB (4-way) → Output HORZ_A/B, VERT_A/B
  └─ RECT → Stage 2.5 (binary HORZ vs VERT, robusto) → Output HORZ, VERT
```

**Diferença de Stage 2.5 vs Stage 3-RECT:**
- Stage 3-RECT: Treinou apenas com RECT limpos
- Stage 2.5: Treina com **RECT + noise (20% AB + 10% SPLIT)**
- Objetivo: Robustez desde o início

**Protocolo Experimental:**

```bash
# 1. Criar dataset Stage 2.5 (RECT + noise)
python3 016_prepare_stage25_dataset.py \
  --rect-dataset v6_dataset_stage3/RECT/block_16 \
  --ab-dataset v6_dataset_stage3/AB/block_16 \
  --noise-ratio 0.3 \
  --output v6_dataset_stage25/block_16

# 2. Treinar Stage 2.5
python3 017_train_stage25.py \
  --dataset-dir v6_dataset_stage25/block_16 \
  --epochs 30 \
  --output-dir stage25

# 3. Modificar pipeline para usar Stage 2.5
# (modificar 008_run_pipeline_eval_v6.py)
```

**Custo:**
- 2 dias implementação (dataset + stage 2.5)
- 3 dias treinamento + tuning
- 2 dias pipeline integration
- **Total:** 7 dias

**Ganho Esperado:**
- Stage 2.5 accuracy: 60-70% (melhor que Stage 3-RECT 4.49%)
- Pipeline accuracy: +0.3-0.8pp

**Literatura:**
- Bengio et al., 2009: Curriculum Learning
- Kumar et al., 2010: Self-Paced Learning for Latent Variable Models

---

### Resumo Fase 3

| Técnica | Custo | Ganho Esperado | Risco |
|---------|-------|----------------|-------|
| **3.1 Multi-Task Stage 2** | 10 dias | +0.5-1.0pp | 🟡 Médio |
| **3.2 ViT Stage 2** | 14 dias | +0.5-1.5pp | 🔴 Alto |
| **3.3 Stage 2.5** | 7 dias | +0.3-0.8pp | 🟢 Baixo |

**Recomendação Fase 3:**
1. Tentar 3.3 (Stage 2.5) primeiro - menor risco
2. Se falhar, tentar 3.1 (Multi-Task) - médio risco, alto impacto
3. Evitar 3.2 (ViT) - alto risco, incerto com poucos dados

---

## 📊 Tabela Resumo: Todas Opções

| ID | Técnica | Fase | Custo | Ganho | Prioridade |
|----|---------|------|-------|-------|------------|
| **1.1** | Diagnose Stage 3-RECT | 1 | 2h | +0.1-0.3pp | 🔴🔴🔴 |
| **1.2** | Threshold Grid Search | 1 | 2h | +0.1-0.3pp | 🔴🔴🔴 |
| **1.3** | Stage 2 Strong Aug | 1 | 1d | +0.2-0.4pp | 🔴🔴 |
| **2.1** | Noise Injection Stage 3 | 2 | 3d | +0.3-0.6pp | 🔴🔴 |
| **2.2** | Focal Loss Tuning | 2 | 2d | +0.2-0.4pp | 🔴 |
| **2.3** | Ensemble Frozen+Scratch | 2 | 7d | +0.5-1.0pp | 🟡 |
| **3.1** | Multi-Task Stage 2 | 3 | 10d | +0.5-1.0pp | 🟡 |
| **3.2** | ViT Stage 2 | 3 | 14d | +0.5-1.5pp | 🟢 |
| **3.3** | Stage 2.5 Intermediate | 3 | 7d | +0.3-0.8pp | 🔴 |

**Legenda Prioridade:**
- 🔴 Alta (recomendado)
- 🟡 Média (se necessário)
- 🟢 Baixa (exploratório)

---

## 🎯 Recomendação Estratégica Final

### Cenário 1: Objetivo = Fechar Gap (-0.34pp) Rapidamente
**Executar:** Fase 1 (3-4 dias)
- 1.1 + 1.2 (4h total): Diagnóstico + Threshold
- 1.3 (1 dia): Strong augmentation
- **Probabilidade de sucesso:** 60-70%

### Cenário 2: Objetivo = Alcançar 48.5-49% (Otimização Moderada)
**Executar:** Fase 1 + Fase 2 (7-10 dias)
- Fase 1 completa (4 dias)
- 2.1 Noise Injection (3 dias)
- **Probabilidade de sucesso:** 70-80%

### Cenário 3: Objetivo = Push State-of-the-Art (>49%)
**Executar:** Fase 1 + Fase 2 + Fase 3 (3-4 semanas)
- Fase 1 + Fase 2 (7-10 dias)
- 3.1 Multi-Task (10 dias) OU 3.3 Stage 2.5 (7 dias)
- **Probabilidade de sucesso:** 50-60%

---

## ⚠️ Considerações para Tese

### Trade-off: Tempo vs Performance

| Decisão | Tempo | Accuracy Final | Contribuição Científica |
|---------|-------|----------------|-------------------------|
| **Aceitar 47.66%** | 0 dias | 47.66% | ✅ Boa (trade-off standalone vs pipeline) |
| **Fase 1** | 3-4 dias | 48.0-48.4% | ✅✅ Excelente (validação completa) |
| **Fase 1+2** | 7-10 dias | 48.5-49.0% | ✅✅✅ Ótima (técnicas avançadas) |
| **Todas Fases** | 3-4 sem | 49.0-50.0% | ✅✅✅ Excepcional (arquitetura inovadora) |

### Recomendação para PhD

**FASE 1 é suficiente para tese:**
- Demonstra rigor científico (diagnóstico, otimização, validação)
- Tempo razoável (3-4 dias)
- Alta probabilidade de alcançar meta (48%)
- Permite focar em escrita da tese

**FASE 2 adiciona valor moderado:**
- Técnicas mais sofisticadas (noise injection, ensemble)
- Tempo ainda razoável (1-1.5 semanas)
- Recomendado SE Fase 1 falhar OU se há tempo disponível

**FASE 3 é exploratória:**
- Alto risco, alto custo (1-2 semanas)
- Contribuição científica marginal (arquitetura não é foco da tese)
- Recomendado APENAS se tese tem capítulo sobre arquiteturas avançadas

---

## 📅 Cronograma Sugerido

### Semana 1 (Dias 1-4): FASE 1
- **Dia 1 manhã:** 1.1 Diagnose Stage 3-RECT (2h)
- **Dia 1 tarde:** 1.2 Threshold grid search (2h)
- **Dia 2-3:** 1.3 Stage 2 strong augmentation (1.5 dias)
- **Dia 4:** Pipeline re-evaluation + análise

**Checkpoint:** Accuracy ≥ 48%?
- ✅ SIM: PARAR, documentar, commit
- ❌ NÃO: Avaliar se continua para Fase 2

### Semana 2 (Dias 5-11): FASE 2 (se necessário)
- **Dia 5-7:** 2.1 Noise injection Stage 3 (3 dias)
- **Dia 8-9:** 2.2 Focal loss tuning (2 dias)
- **Dia 10-11:** Pipeline re-evaluation + análise + decisão

**Checkpoint:** Accuracy ≥ 48.5%?
- ✅ SIM: PARAR, documentar
- ❌ NÃO: Avaliar Fase 3 ou aceitar resultado

---

## 📖 Referências Completas

### Fase 1
1. Zhang et al., 2018: mixup - Beyond Empirical Risk Minimization
2. Yun et al., 2019: CutMix - Regularization Strategy to Train Strong Classifiers
3. Davis & Goadrich, 2006: Precision-Recall and ROC Curves
4. Johnson & Khoshgoftaar, 2019: Survey on Deep Learning with Class Imbalance

### Fase 2
5. Hendrycks et al., 2019: Using Pre-Training Can Improve Model Robustness
6. Lin et al., 2017: Focal Loss for Dense Object Detection
7. Dietterich, 2000: Ensemble Methods in Machine Learning
8. Cui et al., 2019: Class-Balanced Loss

### Fase 3
9. Caruana, 1997: Multitask Learning
10. Dosovitskiy et al., 2021: An Image is Worth 16×16 Words (ViT)
11. Bengio et al., 2009: Curriculum Learning
12. Ruder, 2017: An Overview of Multi-Task Learning

---

**Última Atualização:** 13 de outubro de 2025  
**Próximo Passo:** Decisão - executar Fase 1 ou aceitar resultado atual (47.66%)
