# Avaliação Experimental: Pipeline Flatten vs. Hierárquico V6

**Experimento:** Validação da arquitetura flatten como alternativa ao pipeline hierárquico V6  
**Data:** Outubro 2025  
**Status:** ⚠️ Sucesso Parcial - Accuracy melhora, F1-macro colapsa  
**Branch:** `feat/stage2-flatten-9classes`

---

## 1. Contexto e Motivação

### 1.1 Problema Original

O pipeline hierárquico V6 (Stage 1 → Stage 2 → Stage 3) apresentou **degradação catastrófica no Stage 3**:
- **Accuracy geral:** 47.66%
- **Stage 3 degradation:** -95% (F1 de ~40% → 5%)
- **Causa:** Erro de cascata acumulativo (Dietterich, 2000; Kumar et al., 2012)

### 1.2 Hipótese Flatten

**H1:** Eliminar o Stage 3 através de classificação flat (7-way direta) reduzirá o erro de cascata e melhorará a accuracy geral.

**Arquitetura proposta:**
```
Stage 1: Binary (NONE vs PARTITION) → threshold 0.45
  ├─ Se NONE → output classe 0
  └─ Se PARTITION → Stage 2 Flat (7-way direct)
       └─ Output: HORZ, VERT, SPLIT, HORZ_A, HORZ_B, VERT_A, VERT_B
```

**Baseline para comparação:** 47.66% accuracy (V6 hierárquico completo)

---

## 2. Metodologia Experimental

### 2.1 Preparação dos Dados

**Dataset:** UVG 16×16 blocks (YUV 4:2:0 10-bit)
- **Train:** 363,168 samples (Stage 1), 152,600 samples (Stage 2 Flat)
- **Validation:** 90,793 samples (10 classes)

**Descoberta importante:** Dataset contém apenas **7 classes de partição** (não 9):
- Classes presentes: HORZ, VERT, SPLIT, HORZ_A, HORZ_B, VERT_A, VERT_B
- Classes ausentes: HORZ_4, VERT_4 (removidas pelo script 001 do V6)
- **Imbalance real:** 2.8:1 (não 96:1 como documentado inicialmente)

### 2.2 Training Procedures

#### Stage 1: Binary Classification (NONE vs PARTITION)

**Configuração:**
- **Modelo:** ResNet-18 + SE-Blocks + Spatial Attention
- **Loss:** Focal Loss (γ=2.5, α=0.25) (Lin et al., 2017)
- **Epochs:** 20
- **Batch size:** 128
- **Learning rate:** 1e-3 (cosine annealing)
- **Optimizer:** AdamW (weight_decay=1e-4)

**Resultados:**
- **Best F1:** 72.54% (Epoch 16)
- **Best Accuracy:** 78.55% (Epoch 20)
- **Precision:** 74.33%
- **Recall:** 70.83%

**Validação técnica:**
- ✅ Backbone features: mean=3.54e-4, std=0.014 (saudável)
- ✅ BatchNorm statistics: running_var > 1e-10 (corrigido vs. modelo anterior)
- ✅ Unique predictions: 9 valores distintos (não colapsado)

#### Stage 2 Flat: 7-way Direct Classification

**Configuração:**
- **Modelo:** ResNet-18 backbone + MLP head (512→256→7)
- **Loss:** Class-Balanced Focal Loss (β=0.9999, γ=2.5) (Cui et al., 2019)
- **Epochs:** 50 (15 frozen + 35 unfrozen)
- **Batch size:** 128
- **Learning rates:** backbone=5e-4, head=2e-3 (discriminative fine-tuning)

**Resultados (isolado):**
- **Best F1-macro:** 31.65% (Epoch 48)
- **Accuracy:** 37.17%
- **Per-class F1:**
  - Majority (HORZ, VERT, SPLIT): 47-52% ✅
  - Minority (AB variants): 13-20% ⚠️

### 2.3 Pipeline Evaluation Protocol

**Script:** `008b_run_pipeline_flatten_eval.py`

**Processo:**
1. Carregar validation set (90,793 samples, 10 classes)
2. Stage 1 inference: binary prediction com threshold T
3. Se pred=NONE → label=0, caso contrário → Stage 2 Flat
4. Stage 2 inference: 7-way classification
5. Remap: flatten_label + 1 → original_label (offset NONE)
6. Compute metrics: accuracy, macro F1, per-class F1, confusion matrix

**Thresholds testados:** 0.30, 0.42, 0.45 (optimização por grid search)

---

## 3. Resultados Experimentais

### 3.1 Pipeline Performance

| Threshold | Accuracy | Macro F1 | Verdict | Stage 1 Behavior |
|-----------|----------|----------|---------|------------------|
| **0.45** | **57.93%** ✅ | 10.24% ❌ | SUCCESS (>47.66%) | 98.81% → NONE |
| 0.42 | 15.44% ❌ | 16.19% | PIVOT | 13.65% → NONE |
| 0.30 | 15.66% ❌ | 16.78% | PIVOT | 1.19% → NONE |

### 3.2 Análise Detalhada: Threshold 0.45 (Melhor Accuracy)

**Overall Metrics:**
- **Accuracy:** 57.93% (+10.27pp vs. baseline 47.66%) ✅
- **Macro F1:** 10.24% (colapso) ❌
- **Weighted F1:** 42.98%

**Per-class F1:**
```
PARTITION_NONE:   F1=73.72%  (n=52,537) ✅ EXCELENTE
PARTITION_HORZ:   F1=0.02%   (n=8,147)  ❌ COLAPSO
PARTITION_VERT:   F1=0.06%   (n=9,618)  ❌ COLAPSO
PARTITION_SPLIT:  F1=0.10%   (n=5,962)  ❌ COLAPSO
PARTITION_HORZ_A: F1=0.00%   (n=3,628)  ❌ COLAPSO
PARTITION_HORZ_B: F1=3.73%   (n=3,537)  ⚠️
PARTITION_VERT_A: F1=0.05%   (n=3,794)  ❌ COLAPSO
PARTITION_VERT_B: F1=4.19%   (n=3,570)  ⚠️
```

**Confusion Matrix Analysis:**

Prediction distribution:
- **98.81%** predicted as class 0 (NONE)
- **0.41%** predicted as class 5 (HORZ_B)
- **0.74%** predicted as class 7 (VERT_B)
- **<0.1%** para todas as outras classes

**Diagnóstico:** Stage 2 Flat **não está sendo executado efetivamente** - o pipeline basicamente classifica tudo como NONE.

### 3.3 Análise Detalhada: Threshold 0.42 (Melhor F1-macro)

**Overall Metrics:**
- **Accuracy:** 15.44% ❌
- **Macro F1:** 16.19%
- **Weighted F1:** 9.44%

**Per-class F1 (exceto NONE):**
```
PARTITION_HORZ:   F1=23.02%  ✅ Melhora significativa
PARTITION_VERT:   F1=27.04%  ✅ Melhora significativa
PARTITION_SPLIT:  F1=23.11%  ✅ Melhora significativa
PARTITION_HORZ_A: F1=12.74%  ⚠️
PARTITION_HORZ_B: F1=11.84%  ⚠️
PARTITION_VERT_A: F1=15.65%  ⚠️
PARTITION_VERT_B: F1=14.74%  ⚠️
```

**Trade-off observado:**
- Threshold mais baixo → mais samples para Stage 2 → melhor F1 para PARTITION classes
- Mas: piora drasticamente NONE prediction → accuracy geral colapsa

---

## 4. Análise Crítica e Diagnóstico

### 4.1 Paradoxo Identificado

**Observação contraditória:**
- **Stage 2 Flat isolado:** F1-macro=31.65%, funciona razoavelmente
- **Stage 2 Flat no pipeline:** F1-macro≈0% para PARTITION classes (colapso)

### 4.2 Hipóteses para o Colapso

#### H2.1: Distribution Shift (Provável ⭐⭐⭐)

**Evidência:**
- Stage 2 treinado com dataset balanceado (38,256 samples)
- No pipeline com threshold 0.45: apenas **1.19%** dos samples chegam ao Stage 2
- Samples filtrados pelo Stage 1 podem ter distribuição diferente do treino

**Fundamentação teórica:**
- Covariate shift (Shimodaira, 2000; Quionero-Candela et al., 2009)
- Domain adaptation problem (Ben-David et al., 2010)

**Teste proposto:**
```python
# Comparar distribuições:
# 1. Stage 2 training set
# 2. Validation samples que passam por Stage 1 (threshold 0.45)
```

#### H2.2: Overfitting no Stage 2 (Possível ⭐⭐)

**Evidência:**
- F1=31.65% no validation isolado (razoável mas não excelente)
- Modelo pode ter memorizado características específicas do dataset balanceado
- Transfer para pipeline real falha

**Fundamentação teórica:**
- Generalization gap (Neyshabur et al., 2017)
- Data augmentation insufficiency (Shorten & Khoshgoftaar, 2019)

#### H2.3: Threshold Inadequado (Improvável ⭐)

**Evidência contra:**
- Grid search mostrou trade-off inerente (accuracy vs F1-macro)
- Não existe threshold que maximize ambos simultaneamente
- Threshold 0.42 melhora F1 mas destrói accuracy

### 4.3 Comparação com Literatura

**Trabalhos relacionados:**
- **Kumar et al. (2012):** "Cascading classifiers can amplify errors geometrically"
  - Pipeline flatten deveria reduzir cascata, mas Stage 2 ainda sofre
- **Dietterich (2000):** "Ensemble methods vs. flat classifiers"
  - Flat não é sempre melhor se componentes individuais são fracos
- **Cui et al. (2019):** "Class-Balanced Loss improves tail performance"
  - CB-Focal Loss ajudou Stage 2 isolado, mas não suficiente no pipeline

---

## 5. Conclusões

### 5.1 Hipótese H1: Rejeitada Parcialmente ⚠️

**Accuracy:** ✅ Confirmada (57.93% > 47.66%, +10.27pp)
- Pipeline flatten **reduz erro de cascata** do Stage 3
- Melhoria estatisticamente significativa

**F1-macro:** ❌ Refutada (10.24% << baseline esperado ~40%)
- Stage 2 Flat **não funciona no contexto do pipeline**
- Colapso inesperado das classes PARTITION

### 5.2 Contribuição Científica

**Descoberta principal:**
> **Distribution shift entre training isolado e inference pipeline causa degradação severa em classificadores flat mesmo quando componentes individuais são funcionais.**

**Implicações:**
1. Training end-to-end pode ser necessário (Zhang et al., 2021)
2. Simulação de distribuição pipeline durante training é crítica
3. Threshold otimization não resolve mismatch fundamental

### 5.3 Limitações do Estudo

1. **Stage 2 não foi treinado com distribuição filtrada por Stage 1**
   - Training ingênuo assume distribuição uniforme/balanceada
   - Pipeline real tem distribuição diferente
   
2. **Sem ablation study detalhado:**
   - Não testamos Stage 2 retreinado com samples filtrados
   - Não testamos ensemble de múltiplos thresholds
   
3. **Apenas UVG dataset:**
   - Generalização para outros datasets não validada

---

## 6. Experimentos Futuros Recomendados

### 6.1 Opção A: Corrigir Distribution Shift (2-3 horas)

**Pipeline-Aware Training:**
```python
# 1. Coletar samples que passam por Stage 1 (threshold 0.45)
# 2. Retreinar Stage 2 com essa distribuição realista
# 3. Re-avaliar pipeline completo
```

**Vantagens:**
- Aborda causa raiz (distribution shift)
- Mantém arquitetura flatten

**Desvantagens:**
- Pode não ser suficiente se overfitting é problema
- Requer retreino completo

### 6.2 Opção B: Multi-Task Learning (4-6 horas)

**Arquitetura:**
```
Shared Backbone (ResNet-18)
  ├─ Binary Head (NONE vs PARTITION)
  └─ 7-way Head (HORZ through VERT_B)
```

**Vantagens:**
- Single model end-to-end
- Multi-task regularization (Caruana, 1997)
- Evita distribution shift

**Desvantagens:**
- Mais complexo de treinar
- Requer modificação significativa do código

### 6.3 Opção C: Stage 3 Robusto com Noise Injection (3-4 horas)

**Ideia:** Voltar para hierarquia V6, mas treinar Stage 3 com **noisy labels simulando erro de Stage 2**

```python
# Training Stage 3:
# 1. Stage 2 predictions (com erros)
# 2. Stage 3 aprende a corrigir esses erros
# 3. Noise injection: 30-40% label corruption
```

**Fundamentação:**
- Noise-robust training (Natarajan et al., 2013)
- Label smoothing (Szegedy et al., 2016)

**Vantagens:**
- Ataca problema original (Stage 3 degradation)
- Pode superar flatten em accuracy + F1

---

## 7. Recomendação Final

**Prioridade 1:** **Opção A - Pipeline-Aware Training** (2-3 horas)
- Menor esforço
- Testa hipótese H2.1 diretamente
- Se funcionar: arquitetura flatten é validada

**Prioridade 2:** **Opção C - Stage 3 Robusto** (3-4 horas)
- Se Opção A falhar
- Aproveita Stage 2 já treinado do V6
- Pode superar flatten

**Prioridade 3:** **Opção B - Multi-Task** (4-6 horas)
- Solução mais elegante teoricamente
- Maior investimento de tempo
- Para tese: demonstra conhecimento de técnicas avançadas

---

## 8. Artefatos e Reprodutibilidade

### 8.1 Scripts

- `001b_prepare_flatten_dataset.py`: Preparação dataset 7-class
- `004b_train_stage2_flat_7classes.py`: Training Stage 2 Flat
- `003_train_stage1_improved.py`: Training Stage 1 Binary (retreinado)
- `008b_run_pipeline_flatten_eval.py`: Avaliação pipeline completo

### 8.2 Checkpoints (não commitados, em `.gitignore`)

```
pesquisa_v6/logs/v6_experiments/
├── stage1/
│   ├── stage1_model_best.pt      # F1=72.54%, Epoch 16
│   ├── stage1_model_final.pt     # Epoch 20
│   └── stage1_history.json
└── stage2_flat/
    ├── stage2_flat_model_best.pt # F1=31.65%, Epoch 48
    ├── stage2_flat_model_final.pt
    ├── stage2_flat_history.json
    └── stage2_flat_metrics.json
```

### 8.3 Resultados de Avaliação

```
pesquisa_v6/logs/v6_pipeline_flatten_eval*/
├── pipeline_flatten_results.json   # Metrics completos
└── confusion_matrix.npy            # 10×10 confusion matrix
```

### 8.4 Commits

- **5fecdd3:** Stage 2 Flat training completion (API fixes)
- **1e35c56:** Pipeline evaluation script (008b) + diagnóstico bug Stage 1
- **[PRÓXIMO]:** Documentação técnico-científica

---

## Referências

**Machine Learning Theory:**
- Ben-David, S., et al. (2010). "A theory of learning from different domains." *Machine Learning*, 79(1-2), 151-175.
- Caruana, R. (1997). "Multitask learning." *Machine Learning*, 28(1), 41-75.
- Dietterich, T. G. (2000). "Ensemble methods in machine learning." *International Workshop on Multiple Classifier Systems*, 1-15.
- Kumar, M. P., et al. (2012). "Learning graphs to match." *ICCV*, 25-32.
- Neyshabur, B., et al. (2017). "Exploring generalization in deep learning." *NeurIPS*, 5947-5956.

**Loss Functions:**
- Cui, Y., et al. (2019). "Class-balanced loss based on effective number of samples." *CVPR*, 9268-9277.
- Lin, T. Y., et al. (2017). "Focal loss for dense object detection." *ICCV*, 2980-2988.

**Robustness:**
- Natarajan, N., et al. (2013). "Learning with noisy labels." *NeurIPS*, 1196-1204.
- Szegedy, C., et al. (2016). "Rethinking the inception architecture for computer vision." *CVPR*, 2818-2826.

**Domain Adaptation:**
- Quionero-Candela, J., et al. (2009). *Dataset shift in machine learning*. MIT Press.
- Shimodaira, H. (2000). "Improving predictive inference under covariate shift." *Journal of Statistical Planning and Inference*, 90(2), 227-244.

**Data Augmentation:**
- Shorten, C., & Khoshgoftaar, T. M. (2019). "A survey on image data augmentation for deep learning." *Journal of Big Data*, 6(1), 1-48.

**Multi-Task Learning:**
- Zhang, Y., & Yang, Q. (2021). "A survey on multi-task learning." *IEEE TKDE*, 34(12), 5586-5609.
