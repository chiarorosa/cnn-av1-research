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
---

## 🎯 Objetivos e Metas

### Métricas Alvo (block_16)

| Componente | Meta Fase 1 | Meta Fase 2 | v6 Obtido (13/10) | Status |
|------------|-------------|-------------|-------------------|--------|
| **Stage 1 F1** | 68-70% | 72-75% | ✅ **72.28%** | Meta Fase 2 atingida |
| **Stage 1 Precisão** | 62-65% | 68-72% | ✅ **67.13%** | Meta Fase 1 atingida |
| **Stage 2 Macro F1** | 45-50% | 55-60% | ✅ **46.51%** (frozen) | Meta Fase 1 atingida |
| **Stage 3-RECT F1** | 75-78% | 80-83% | ⚠️ **68.44%** | **-6.6pp abaixo da meta** |
| **Stage 3-AB F1** | 45-50% | 60-65% | ⚠️ **24.50%** | **-20.5pp abaixo da meta** |
| **Acurácia Final** | 48-52% | 58-63% | ⚠️ **47.66%** | **-0.34pp abaixo da meta** |

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

## 🎯 ANÁLISE DOS PROBLEMAS CRÍTICOS (Baseado em docs_v6/)

### Problema Central: Erro em Cascata Severo

**Diagnóstico (doc 05_avaliacao_pipeline_completo.md):**

| Componente | F1 Standalone | Accuracy Pipeline | Degradação |
|------------|---------------|-------------------|------------|
| Stage 3-RECT | 68.44% | **4.49%** | **-93.4%** ❌❌❌ |
| Stage 3-AB | 24.50% | **1.51%** | **-93.8%** ❌❌❌ |

**Classes Colapsadas no Pipeline:**
- HORZ: 0% F1 (9,618 samples preditos como 0)
- HORZ_A: 0% F1
- VERT_A: 0% F1
- VERT_B: 0% F1

**Root Cause Identificado:** Stage 2 confunde sistematicamente RECT vs AB

### Hipóteses Testadas e REJEITADAS

#### ❌ H2.1: Distribution Shift (doc 08_pipeline_aware_training.md)

**Hipótese:**
> "Stage 2 colapsa porque treinou com distribuição balanceada mas recebe distribuição filtrada por Stage 1."

**Teste:**
- Retreinar Stage 2 com 3,890 samples filtrados por Stage 1 (threshold 0.45)
- **Resultado:** F1=6.74% (pior que baseline 31.65%) - **degradação de -78.7%**

**Conclusão:** REJEITADA - Distribution shift NÃO é a causa primária

**Root causes reais identificados:**
1. Dataset insuficiente (3,890 / 11M params = 1:2,900 ratio)
2. Negative transfer (Stage 1 binary features ≠ Stage 2 multi-class)
3. Architectural flaw (objetivos conflitantes)

---

## 🎯 PRÓXIMOS PASSOS: Resolver Erro em Cascata

### Estratégia Baseada em Evidências
> "Focar no problema real: Stage 2 confunde RECT vs AB, causando colapso dos Stage 3."

### Fase 1: Diagnóstico Profundo (1-2 dias) - **CRÍTICO**

#### 1.1 Analisar Confusão RECT vs AB no Stage 2 🔴 **CRÍTICO**

**Problema Documentado (doc 05):**
```
Ground Truth: HORZ (RECT)
    ↓
Stage 2 Frozen classifica como: AB (ERRADO!)
    ↓
Envia para Stage 3-AB (que nunca viu HORZ!)
    ↓
Stage 3-AB colapsa → prediz HORZ_B (default)
    ↓
Resultado: HORZ_B (ERRADO!)
```

**Protocolo:**
```bash
# Script: pesquisa_v6/scripts/009_analyze_stage2_confusion.py

# 1. Carregar Stage 2 frozen model
# 2. Inferir validation set completo
# 3. Analisar confusion matrix RECT vs AB
# 4. Identificar padrões visuais que causam confusão
# 5. Calcular % de RECT enviado erroneamente para Stage 3-AB
```

**Métricas-chave:**
- Precision RECT (Stage 2): Quantos "RECT" são realmente RECT?
- Recall AB (Stage 2): Quantos AB são corretamente identificados?
- Taxa de "RECT → AB error": % de RECT enviados para Stage 3-AB

**Ganho esperado:** Entendimento claro do erro cascata

#### 1.2 Avaliar Viés do Stage 3-RECT (2h) � **CRÍTICO**

**Problema Observado:** HORZ colapsou (0% F1), VERT superestimado (+16.19%)

**Hipótese:** Stage 3-RECT tem viés extremo para VERT

**Protocolo:**
```bash
# Script: pesquisa_v6/scripts/010_diagnose_stage3_rect.py

python3 010_diagnose_stage3_rect.py \
  --model logs/v6_experiments/stage3_rect/stage3_rect_model_best.pt \
  --dataset v6_dataset_stage3/RECT/block_16/val.pt

# Análises:
# 1. Confusion matrix standalone (HORZ vs VERT)
# 2. F1 per-class standalone  
# 3. Distribuição de probabilidades
# 4. Class distribution no dataset de treino
```

**Ações baseadas em resultado:**
- Se HORZ F1 < 60% standalone → Retreinar com class weights
- Se dataset desbalanceado (VERT > 60%) → Rebalancear com sampler

**Ganho esperado:** +0.1-0.3pp (se viés confirmado e corrigido)

### Fase 2: Soluções Robustas (3-5 dias) - **RECOMENDADA**

**Objetivo:** Resolver confusão RECT vs AB + Robustez dos Stage 3

#### 2.1 Noise Injection em Stage 3 (3 dias) � **ALTA PRIORIDADE**

**Problema Fundamental:**
> "Stage 3-RECT e Stage 3-AB foram treinados apenas com samples CORRETOS. No pipeline, recebem samples ERRADOS do Stage 2 e colapsam."

**Solução:** Adversarial Training / Noise Injection
- Treinar Stage 3 com 20-30% "dirty samples"
- Simula distribuição real que Stage 3 receberá no pipeline
- Modelo aprende robustez a erros do Stage 2

**Stage 3-RECT Robusto:**
```python
# Durante treinamento
for epoch in range(epochs):
    for batch in dataloader_RECT:
        x_rect, y_rect = batch
        
        # 20-30% das vezes, injetar sample AB
        if np.random.rand() < 0.25:
            idx = np.random.randint(len(dataset_AB))
            x_noise, y_noise = dataset_AB[idx]
            
            # Substituir um sample RECT por AB
            x_rect[0] = x_noise
            y_rect[0] = np.random.choice([0, 1])  # Random label
            
        loss = criterion(model(x_rect), y_rect)
```

**Protocolo:**
```bash
# 1. Retreinar Stage 3-RECT com noise
python3 005_train_stage3_rect.py \
  --noise-injection 0.25 \
  --noise-source AB \
  --noise-source SPLIT \
  --epochs 30 \
  --output-dir logs/v6_experiments/stage3_rect_robust

# 2. Retreinar Stage 3-AB com noise  
python3 006_train_stage3_ab_fgvc.py \
  --noise-injection 0.25 \
  --noise-source RECT \
  --noise-source SPLIT \
  --epochs 30 \
  --output-dir logs/v6_experiments/stage3_ab_robust

# 3. Re-avaliar pipeline
python3 008_run_pipeline_eval_v6.py \
  --stage3-rect-model stage3_rect_robust/model_best.pt \
  --stage3-ab-model stage3_ab_robust/model_best.pt
```

**Fundamentação Teórica:**
- Hendrycks et al., 2019: "Using Pre-Training Can Improve Model Robustness"
- Natarajan et al., 2013: "Learning with Noisy Labels"
- Recht et al., 2019: "Do ImageNet Classifiers Generalize to ImageNet?"

**Ganho Esperado:**
- Stage 3-RECT pipeline accuracy: 4.49% → 15-25% (+234-457%)
- Stage 3-AB pipeline accuracy: 1.51% → 5-10% (+231-562%)
- Overall pipeline: **+1.0-2.5pp** → Accuracy 48.7-50.2% ✅✅

**Custo:**
- 1.5 dia retreino Stage 3-RECT (30 epochs)
- 1.5 dia retreino Stage 3-AB (30 epochs)
- 0.5 dia pipeline evaluation + análise
- **Total:** 3.5 dias

#### 2.2 Melhorar Separação RECT vs AB no Stage 2 (2 dias) 🟡 **MÉDIA PRIORIDADE**

**Problema:** Stage 2 confunde RECT vs AB (causa do erro cascata)

**Solução 1: Contrastive Learning**
- Adicionar contrastive loss (Chen et al., 2020 - SimCLR)
- Força backbone a separar melhor RECT vs AB no espaço de features

```python
# Adicionar ao treinamento Stage 2
contrastive_loss = SimCLR_loss(features_rect, features_ab, temperature=0.5)
total_loss = cb_focal_loss + 0.3 * contrastive_loss
```

**Solução 2: Focal Loss Tuning**
- Testar γ=[2.0, 2.5, 3.0] (mais foco em hard examples)
- Testar α customizado por classe (mais peso em AB/RECT)

**Protocolo:**
```bash
# Grid search
gammas = [2.0, 2.5, 3.0]
for gamma in gammas:
    python3 004_train_stage2_redesigned.py \
      --gamma $gamma \
      --epochs 25 \
      --output-dir stage2_gamma${gamma}

# Avaliar qual melhor separa RECT vs AB
```

**Ganho Esperado:** Stage 2 confusion RECT↔AB reduz 30-40% → +0.5-1.0pp pipeline

**Custo:** 2 dias

### Fase 3: Técnicas Avançadas (1-2 semanas) - **EXPLORATÓRIO**

**Condição:** Fase 2 não atingiu 50% + disponibilidade de tempo

#### 3.1 Multi-Task Learning Stage 2 (5 dias)

**Problema:** Stage 2 não aprende geometria interna de RECT

**Solução:** Dual-head architecture
```python
class MultiTaskStage2(nn.Module):
    def __init__(self):
        self.backbone = ImprovedBackbone()
        
        # Head principal: 3-way (SPLIT, RECT, AB)
        self.head_3way = nn.Linear(512, 3)
        
        # Head auxiliar: 2-way (HORZ, VERT) - apenas para RECT
        self.head_rect_geometry = nn.Linear(512, 2)
        
    def forward(self, x):
        features = self.backbone(x)
        pred_3way = self.head_3way(features)
        pred_rect = self.head_rect_geometry(features)
        return pred_3way, pred_rect

# Loss
loss = cb_focal_3way + 0.5 * cross_entropy_rect_geometry
```

**Vantagens:**
- Backbone aprende features para HORZ vs VERT
- Melhora separação RECT vs AB
- Regularização implícita (Caruana, 1997)

**Ganho esperado:** +0.5-1.0pp

**Custo:** 5 dias

#### 3.2 Stage 2.5 Intermediate (4 dias)

**Problema:** Stage 3-RECT recebe samples ruins e colapsa

**Solução:** Adicionar stage intermediário robusto

```
Stage 2 → RECT → Stage 2.5 (HORZ vs VERT, treinado com noise) → Output
```

**Diferença vs Stage 3-RECT:**
- Stage 3-RECT: Treinou só com RECT limpos
- Stage 2.5: Treina com RECT + 30% noise (AB + SPLIT)

**Ganho esperado:** +0.3-0.8pp

**Custo:** 4 dias

---

## 📊 Resumo de Prioridades

| ID | Técnica | Fase | Custo | Ganho | Prioridade |
|----|---------|------|-------|-------|------------|
| 1.1 | Analisar Confusão RECT vs AB | 1 | 4h | Diagnóstico | 🔴🔴🔴 CRÍTICO |
| 1.2 | Diagnose Stage 3-RECT Viés | 1 | 2h | +0.1-0.3pp | 🔴🔴🔴 CRÍTICO |
| **2.1** | **Noise Injection Stage 3** | **2** | **3.5d** | **+1.0-2.5pp** | 🔴🔴🔴 **RECOMENDADO** |
| 2.2 | Melhorar RECT vs AB (Stage 2) | 2 | 2d | +0.5-1.0pp | 🔴🔴 Alta |
| 3.1 | Multi-Task Stage 2 | 3 | 5d | +0.5-1.0pp | 🟡 Exploratório |
| 3.2 | Stage 2.5 Intermediate | 3 | 4d | +0.3-0.8pp | 🟡 Exploratório |

---

## 🎯 Recomendação Estratégica Final

### **PLANO RECOMENDADO: Fase 1 + Fase 2 (5-7 dias)**

```
Dia 1: 
  - 1.1 Analisar Confusão RECT vs AB (4h)
  - 1.2 Diagnose Stage 3-RECT (2h)
  - Decisão: Confirmar diagnóstico do erro cascata

Dias 2-4:
  - 2.1 Noise Injection Stage 3-RECT (1.5 dias)
  - 2.1 Noise Injection Stage 3-AB (1.5 dias)
  
Dia 5:
  - Re-avaliar pipeline com modelos robustos
  - Análise de resultados

Dias 6-7 (SE NECESSÁRIO):
  - 2.2 Melhorar RECT vs AB com Focal Loss tuning
```

**Probabilidade de sucesso:** 70-85%  
**Ganho esperado:** +1.0-3.0pp → **Accuracy 48.7-50.7%** ✅✅

**Fundamentação:**
- **Fase 1** confirma diagnóstico (erro cascata Stage 2→3)
- **Fase 2 (2.1)** ataca causa raiz (Stage 3 não robusto a erros)
- Técnica validada na literatura (Hendrycks et al., 2019; Natarajan et al., 2013)
- Implementação relativamente simples
- Risco baixo (pior caso: sem melhoria, mas não piora)

---

## � Ação Imediata Recomendada (HOJE - 13/10)

### **COMEÇAR COM FASE 1: Diagnóstico Profundo**

```bash
# 1. Criar Script 009: Análise de Confusão RECT vs AB (2-3 horas)
cd pesquisa_v6/scripts
# Implementar 009_analyze_stage2_confusion.py
# - Carregar Stage 2 frozen model
# - Inferir validation set
# - Gerar confusion matrix detalhada RECT vs AB
# - Calcular taxa de erro Stage 2 → Stage 3

# 2. Criar Script 010: Diagnose Stage 3-RECT (1-2 horas)
# Implementar 010_diagnose_stage3_rect.py
# - Avaliar Stage 3-RECT standalone
# - Verificar viés VERT vs HORZ
# - Analisar class distribution treino

# 3. Executar diagnósticos (1 hora)
python3 009_analyze_stage2_confusion.py
python3 010_diagnose_stage3_rect.py

# 4. Análise de resultados e decisão (1 hora)
# - Confirmar hipótese de erro cascata
# - Decidir: prosseguir para Fase 2 (Noise Injection)
```

**Meta do dia:** Confirmar diagnóstico e planejar Fase 2

**Próximos passos (14-15/10):**
- Implementar noise injection nos scripts 005 e 006
- Treinar Stage 3-RECT e Stage 3-AB robustos
- Re-avaliar pipeline

---

## �🔧 Melhorias Técnicas Implementadas (v6)

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
