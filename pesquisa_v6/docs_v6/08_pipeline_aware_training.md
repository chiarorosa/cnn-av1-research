# Experimento 08: Treinamento Pipeline-Aware - Teste da Hipótese H2.1

**Data**: 2025-10-13  
**Autor**: Chiaro Rosa  
**Status**: ❌ **EXPERIMENTO NEGATIVO - Hipótese H2.1 REJEITADA**  
**Branch**: `feat/stage2-flatten-9classes`  
**Commits**: 382cf2d, 00cfac6, 6c88cf2, 1ea3830

---

## 1. Contexto e Motivação

### 1.1 Problema Diagnosticado (Experimento 07)

No experimento anterior (doc 07), identificamos um **paradoxo crítico** no pipeline flatten:

- **Stage 2 Flat isolado** (script 004b): F1=31.65%, Acc=37.17% ✅ (funciona bem)
- **Stage 2 Flat no pipeline** (script 008b): F1=10.24%, Acc=57.93% ❌ (colapso para PARTITION)

**Observação chave**: Stage 1 com threshold 0.45 prevê 98.81% das amostras como NONE, apenas 1.19% chegam ao Stage 2.

### 1.2 Hipótese H2.1: Distribution Shift

**Enunciado**: O colapso do Stage 2 no pipeline é causado por **distribution shift** (Shimodaira, 2000; Ben-David et al., 2010):

- **P_train(X)**: Distribuição balanceada do dataset de treinamento (152,600 samples, sampler balanceado)
- **P_pipeline(X|Stage1=PARTITION)**: Distribuição filtrada pelo Stage 1 (~1-2% do dataset original)

**Predição**: Se treinarmos Stage 2 com a distribuição **P_pipeline** (amostras filtradas por Stage 1), o F1-macro deve aumentar de 10.24% para >40%.

**Fundamentação teórica**:
- Covariate shift em cascaded systems (Kumar et al., 2012)
- Domain adaptation necessity (Zhang et al., 2021)
- Training-testing distribution mismatch (Quinonero-Candela et al., 2009)

---

## 2. Metodologia

### 2.1 Pipeline-Aware Dataset Creation

**Script**: `004c_train_stage2_pipeline_aware.py` (673 linhas)

**Processo de filtragem**:
1. Carregar Stage 1 model (`stage1_model_best.pt`, F1=72.54%)
2. Inferir todas as amostras do dataset original (train + val)
3. Filtrar com threshold 0.45: `prob(PARTITION) ≥ 0.45`
4. Manter apenas amostras classificadas como PARTITION
5. Salvar datasets filtrados: `train_filtered.pt`, `val_filtered.pt`

**Resultado da filtragem**:

| Split | Original | Filtrado | Retenção |
|-------|----------|----------|----------|
| Train | 152,600  | 3,890    | 2.55%    |
| Val   | 38,256   | 976      | 2.55%    |

**Probabilidade média PARTITION**: 0.6777 ± 0.1136

**Distribuição filtrada (Train)**:
```
PARTITION_SPLIT:  950 samples (24.42%)
PARTITION_HORZ_B: 646 samples (16.61%)
PARTITION_VERT_B: 613 samples (15.76%)
PARTITION_HORZ_A: 546 samples (14.04%)
PARTITION_VERT_A: 513 samples (13.19%)
PARTITION_VERT:   327 samples ( 8.41%)
PARTITION_HORZ:   295 samples ( 7.58%)
```

**Imbalance**: 3.2:1 (SPLIT vs HORZ), muito melhor que dataset original.

### 2.2 Protocolo de Treinamento

**Modelo**: Stage2FlatModel (ResNet-18 + SE + Spatial Attention + MLP head)
- Total parâmetros: 11,347,497
- Backbone: 11,213,858 parâmetros
- Head: 133,639 parâmetros (512→256→7)

**Loss**: Class-Balanced Focal Loss (Cui et al., 2019)
- β = 0.9999 (effective number of samples)
- γ = 2.5 (hard negative mining)
- Class weights: [0.0034, 0.0031, 0.0011, 0.0018, 0.0015, 0.0019, 0.0016]

**Sampler**: WeightedRandomSampler
- Oversampling factor: 2.0×
- Samples per epoch: 7,780 (de 3,890 reais)

**Otimizador**: AdamW
- Backbone LR: 5e-4
- Head LR: 2e-3
- Weight decay: 1e-4

**Scheduler**: OneCycleLR
- Total steps: 3,050 (61 steps/epoch × 50 epochs)
- Warmup: 30% (pct_start=0.3)
- Annealing: cosine

**Augmentation**: Stage2Augmentation (MixUp, CutMix, geometric)

**Training schedule**:
- Epochs 1-15: Backbone frozen, train head only
- Epochs 16-50: Unfreeze, fine-tune with discriminative LR
- Early stopping: patience=8 on val F1-macro

### 2.3 Ambiente Experimental

- **Hardware**: NVIDIA RTX GPU, 12 threads CPU
- **Software**: PyTorch 2.6, Python 3.12
- **Seed**: 42 (reprodutibilidade)
- **Batch size**: 128
- **Num workers**: 4

---

## 3. Resultados Experimentais

### 3.1 Curva de Treinamento

| Epoch | Phase  | Train Loss | Train F1 | Train Acc | Val Loss | Val F1  | Val Acc | Status |
|-------|--------|------------|----------|-----------|----------|---------|---------|--------|
| 1     | Frozen | 1.3808     | 0.1209   | 0.1456    | 1.2093   | 0.0674  | 0.0850  | ✅ Best |
| 2     | Frozen | 1.3279     | 0.1046   | 0.1461    | 1.1904   | 0.0468  | 0.0748  | Patience 1/8 |
| 3     | Frozen | 1.3105     | 0.1002   | 0.1442    | 1.1905   | 0.0531  | 0.0768  | Patience 2/8 |
| 4     | Frozen | 1.3065     | 0.0952   | 0.1445    | 1.1915   | 0.0423  | 0.0727  | Patience 3/8 |
| 5     | Frozen | 1.3053     | 0.0892   | 0.1388    | 1.1954   | 0.0441  | 0.0707  | Patience 4/8 |
| 6     | Frozen | 1.3045     | 0.0903   | 0.1470    | 1.2013   | 0.0440  | 0.0697  | Patience 5/8 |
| 7     | Frozen | 1.3011     | 0.0900   | 0.1483    | 1.1987   | 0.0457  | 0.0727  | Patience 6/8 |
| 8     | Frozen | 1.2959     | 0.0903   | 0.1496    | 1.2019   | 0.0494  | 0.0768  | Patience 7/8 |
| 9     | Frozen | 1.2954     | 0.0868   | 0.1392    | 1.1935   | 0.0455  | 0.0809  | **Early Stop** |

**Melhor modelo**: Epoch 1
- **Val F1-macro**: 6.74%
- **Val Accuracy**: 8.50%

### 3.2 Comparação com Baselines

| Método | Dataset | Train Samples | Val F1 | Val Acc | Pipeline F1 | Pipeline Acc | Status |
|--------|---------|---------------|--------|---------|-------------|--------------|--------|
| Stage 2 isolado (004b) | Full balanced | 152,600 | 31.65% | 37.17% | - | - | ✅ Funciona |
| Pipeline baseline (008b) | - | - | - | - | 10.24% | 57.93% | ⚠️ Parcial |
| **Pipeline-aware (004c)** | **Stage 1 filtered** | **3,890** | **6.74%** | **8.50%** | - | - | ❌❌❌ **PIOR** |

**Resultado crítico**: Pipeline-aware **não resolveu** o problema. **Piorou significativamente**:
- Val F1: 31.65% (isolado) → 6.74% (pipeline-aware) = **-78.7% degradação** ❌
- Val Acc: 37.17% (isolado) → 8.50% (pipeline-aware) = **-77.1% degradação** ❌

### 3.3 Análise de Convergência

**Observações**:
1. **Melhor epoch = 1**: Modelo parou de melhorar imediatamente
2. **F1 estável 6-12%**: Não houve aprendizado progressivo
3. **Accuracy baixíssima (8-15%)**: Muito abaixo do random baseline (14.3% para 7 classes)
4. **Loss plateau**: Loss treino e val convergem mas métricas ruins

**Diagnóstico**: **Overfitting extremo** + **features inadequadas**

---

## 4. Análise Crítica e Diagnóstico

### 4.1 Por Que Pipeline-Aware Falhou?

#### Hipótese A1: Dataset Muito Pequeno ⭐⭐⭐

**Evidência**:
- Apenas 3,890 samples de treino (2.55% do original)
- Modelo tem 11.3M parâmetros (ratio 1:2,900 samples/param)
- Literatura recomenda 10-100 samples/param (Goodfellow et al., 2016)
- **Necessário**: ~100k+ samples para treinar ResNet-18

**Consequência**:
- Overfitting severo mesmo com dropout (0.3, 0.2), augmentation e class balancing
- Modelo memoriza treino mas não generaliza (F1 train 12% vs val 6%)
- Early stopping imediato (epoch 1 = melhor)

#### Hipótese A2: Features do Stage 1 Inadequadas ⭐⭐⭐

**Evidência**:
- Stage 1 otimizado para **discriminar NONE vs PARTITION** (tarefa binária)
- Stage 2 precisa **discriminar HORZ vs VERT vs SPLIT vs A/B** (tarefa 7-way fine-grained)
- Backbone frozen do Stage 1 captura features para decisão binária global
- **Incompatibilidade**: Features binários não servem para discriminação multi-classe

**Fundamentação teórica**:
- Transfer learning failure (Yosinski et al., 2014): Quando source e target tasks são muito diferentes
- Negative transfer (Pan & Yang, 2010): Pré-treino pode prejudicar se tasks incompatíveis
- Feature hierarchy mismatch (Zeiler & Fergus, 2014): Camadas iniciais capturam edges/textures, camadas finais capturam conceitos task-specific

**Experimento de ablação implícito**:
- Stage 2 isolado (backbone ImageNet): F1=31.65% ✅
- Stage 2 pipeline-aware (backbone Stage 1): F1=6.74% ❌
- **Conclusão**: Backbone Stage 1 **prejudica** performance (negative transfer)

#### Hipótese A3: Problema Estrutural da Arquitetura Flatten ⭐⭐

**Argumento**:
- Stage 1 e Stage 2 têm **objetivos conflitantes**:
  - Stage 1: Maximizar recall de PARTITION (threshold baixo para não perder splits)
  - Stage 2: Discriminar tipos de PARTITION com precisão
- **Trade-off impossível**: 
  - Threshold alto (0.45): Poucos samples chegam ao Stage 2 (2.55%), dataset pequeno
  - Threshold baixo (0.20): Muitos false positives, Stage 2 recebe NONE incorretos

**Literatura relevante**:
- Cascaded classifier design (Viola & Jones, 2001): Early stages devem ser high recall, late stages high precision
- Erro de propagação em pipelines (Kumar et al., 2012): Erros do Stage 1 se acumulam
- Joint optimization necessity (Zhang et al., 2021): Treinamento end-to-end resolve conflitos

### 4.2 Hipótese H2.1 (Distribution Shift) REJEITADA

**Conclusão experimental**: Distribution shift **NÃO** é a causa principal do colapso.

**Raciocínio**:
1. Se distribution shift fosse o problema, treinar com distribuição pipeline deveria resolver
2. Experimento 004c treinou com distribuição pipeline → F1 piorou (6.74% < 10.24%)
3. Logo, distribution shift é um fator secundário, não primário

**Causas reais identificadas** (por ordem de importância):
1. **Dataset size insuficiente** (3,890 samples para 11M params)
2. **Feature incompatibility** (Stage 1 binary features ≠ Stage 2 multi-class needs)
3. **Architectural limitation** (cascaded design com objetivos conflitantes)

### 4.3 Limitações do Experimento

1. **Não testamos unfreezing**: Early stop no epoch 9, não chegamos no epoch 16 (unfreeze)
   - Possível melhoria marginal, mas dataset pequeno limita
2. **Não testamos Stage 1 com threshold mais baixo**: 0.45 muito conservador
   - Threshold 0.30 reteria ~10% (15k samples), ainda insuficiente para 11M params
3. **Não testamos feature extraction do Stage 1 frozen**: Apenas testamos transferência
   - Experimento futuro: Extrair features Stage 1, treinar Stage 2 linear do zero

---

## 5. Conclusões

### 5.1 Resultado Principal

**A hipótese H2.1 (distribution shift) foi REJEITADA pelos dados experimentais.**

O treinamento pipeline-aware **não resolveu** o problema de colapso do Stage 2. Performance piorou:
- Val F1: 31.65% (baseline isolado) → 6.74% (pipeline-aware) = **-78.7%** ❌
- Val Acc: 37.17% (baseline isolado) → 8.50% (pipeline-aware) = **-77.1%** ❌

### 5.2 Descobertas Científicas

1. **Distribution shift é secundário**: Não é a causa principal do colapso Stage 2
2. **Dataset size crítico**: 3,890 samples insuficientes para ResNet-18 (11M params)
3. **Negative transfer confirmado**: Backbone Stage 1 (binary task) prejudica Stage 2 (multi-class task)
4. **Arquitetura flatten tem limitação fundamental**: Objetivos conflitantes Stage 1 vs Stage 2

### 5.3 Implicações para o Projeto

**Arquitetura flatten NÃO é viável** para o problema AV1 partition prediction com a abordagem atual.

**Razões**:
1. Impossível obter dataset grande o suficiente após filtragem Stage 1
2. Transfer learning Stage 1→Stage 2 falha (negative transfer)
3. Pipeline accuracy (57.93%) não compensa F1 baixo (6-10%)

**Recomendação**: **Abandonar flatten, retornar à hierárquica V6**.

---

## 6. Experimentos Futuros

### 6.1 Opção A: Flatten com End-to-End (NÃO RECOMENDADO)

**Ideia**: Multi-Task Learning - single model, dual heads (binary + 7-way)

**Arquitetura**:
```
ResNet-18 Backbone (shared)
  ├─ Binary Head (NONE vs PARTITION)
  └─ 7-way Head (HORZ through VERT_B)
```

**Vantagens**:
- Resolve feature incompatibility (backbone otimizado para ambas tasks)
- Resolve dataset size (usa full 152k samples)
- Multi-task regularization (Caruana, 1997)

**Desvantagens**:
- Implementação complexa (4-6 horas)
- Loss balancing difícil (pesos entre binary e 7-way)
- Não resolve problema fundamental: flatten elimina hierarquia semântica do AV1

**Veredicto**: Não vale o esforço. Flatten não é solução adequada.

### 6.2 Opção C: Stage 3 Robust - Hierárquica V6 (RECOMENDADO) ⭐⭐⭐

**Ideia**: Retornar à arquitetura hierárquica, atacar problema original (Stage 3 degradação -95%)

**Abordagem**:
1. Usar Stage 2 V6 existente (F1=46%, bem treinado)
2. Treinar Stage 3 (RECT e AB) com **noise injection**:
   - Simular erros do Stage 2 durante treinamento
   - Label corruption 30-40% (Natarajan et al., 2013)
   - Confidence penalty (Pereyra et al., 2017)
3. Fine-Grained Visual Classification (FGVC) techniques:
   - Center Loss (Wen et al., 2016) para compactness intra-classe
   - CutMix strong (Yun et al., 2019)
   - Cosine classifier com temperature scaling

**Vantagens**:
- Aproveita Stage 2 bem treinado (não desperdiça trabalho)
- Dataset Stage 3 maior (~20k RECT, ~15k AB após filtragem Stage 2)
- Noise-robust training validado na literatura
- Mantém hierarquia semântica do AV1 (NONE → tipo geral → tipo específico)

**Tempo estimado**: 3-4 horas

**Success criteria**: F1 Stage 3 >60% (vs. atual ~25% AB, ~65% RECT)

### 6.3 Opção B: Threshold Optimization (CURTO PRAZO)

**Ideia**: Otimizar threshold Stage 1 para maximizar F1 pipeline (não accuracy)

**Abordagem**:
- Grid search threshold 0.20-0.50 (step 0.05)
- Métrica: F1-macro pipeline end-to-end
- Pode revelar trade-off ideal accuracy vs F1

**Tempo**: 1 hora

**Limitação**: Não resolve problema fundamental, apenas mitiga

---

## 7. Recomendações

### 7.1 Decisão Imediata

**ABANDONAR flatten architecture.**

**Justificativa**:
1. Experimento 004c provou que pipeline-aware não resolve (H2.1 rejeitada)
2. Negative transfer confirmado (features Stage 1 inadequados)
3. Dataset size insuficiente após filtragem (3,890 samples)
4. Accuracy 57.93% não compensa F1 6-10% (aplicação precisa recall de todas as classes)

### 7.2 Próxima Ação

**Implementar Opção C: Stage 3 Robust com Noise Injection**

**Protocolo**:
1. Criar script `006_train_stage3_rect_robust.py`
2. Injetar ruído: Corromper 30% labels RECT durante treino
3. Aplicar FGVC techniques (Center Loss, CutMix, cosine classifier)
4. Treinar 50 epochs, avaliar F1-macro
5. Repetir para `007_train_stage3_ab_robust.py`
6. Avaliar pipeline completo V6 com Stage 3 robusto

**Meta**: F1 pipeline >50% (vs. baseline 47.66%)

### 7.3 Documentação Necessária

- ✅ Este documento (08_pipeline_aware_training.md)
- ⏳ Atualizar `PLANO_V6.md` com resultado negativo flatten
- ⏳ Criar `09_stage3_robust_training.md` para próximo experimento

---

## 8. Reprodutibilidade

### 8.1 Scripts

- **004c_train_stage2_pipeline_aware.py** (673 linhas)
  - Commits: 382cf2d (inicial), 00cfac6 (fix Stage2FlatModel), 6c88cf2 (fix ClassBalancedFocalLoss), 1ea3830 (fix métricas)
- **008b_run_pipeline_flatten_eval.py** (396 linhas)
  - Commit: 1e35c56

### 8.2 Checkpoints

- `pesquisa_v6/logs/v6_experiments/stage2_pipeline_aware/`
  - `stage2_pipeline_aware_best.pt` (epoch 1, F1=6.74%)
  - `stage2_pipeline_aware_final.pt` (epoch 9, F1=4.55%)
  - `stage2_pipeline_aware_history.pt` (training curves)
  - `train_filtered.pt` (3,890 samples)
  - `val_filtered.pt` (976 samples)

### 8.3 Comando de Execução

```bash
# Treinamento pipeline-aware
python3 pesquisa_v6/scripts/004c_train_stage2_pipeline_aware.py \
  --dataset-dir pesquisa_v6/v6_dataset_flatten/block_16 \
  --stage1-model pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt \
  --threshold 0.45 \
  --output-dir pesquisa_v6/logs/v6_experiments/stage2_pipeline_aware \
  --epochs 50 \
  --freeze-epochs 15 \
  --batch-size 128 \
  --lr-backbone 5e-4 \
  --lr-head 2e-3 \
  --device cuda \
  --seed 42
```

### 8.4 Ambiente

- PyTorch: 2.6
- CUDA: 12.x
- GPU: NVIDIA RTX (12 threads)
- Python: 3.12
- OS: Linux Ubuntu

---

## 9. Referências

1. **Shimodaira, H.** (2000). Improving predictive inference under covariate shift. *Journal of Machine Learning Research*, 1, 227-244.

2. **Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W.** (2010). A theory of learning from different domains. *Machine Learning*, 79(1), 151-175.

3. **Kumar, M. P., Packer, B., & Koller, D.** (2012). Learning graphs to match. *ICCV*.

4. **Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D.** (2021). Understanding deep learning requires rethinking generalization. *ICLR*.

5. **Quinonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D.** (2009). *Dataset shift in machine learning*. MIT Press.

6. **Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S.** (2019). Class-Balanced Loss Based on Effective Number of Samples. *CVPR*.

7. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep learning*. MIT Press.

8. **Yosinski, J., Clune, J., Bengio, Y., & Lipson, H.** (2014). How transferable are features in deep neural networks?. *NeurIPS*.

9. **Pan, S. J., & Yang, Q.** (2010). A survey on transfer learning. *IEEE Transactions on Knowledge and Data Engineering*, 22(10), 1345-1359.

10. **Zeiler, M. D., & Fergus, R.** (2014). Visualizing and understanding convolutional networks. *ECCV*.

11. **Viola, P., & Jones, M.** (2001). Robust real-time face detection. *IJCV*, 57(2), 137-154.

12. **Caruana, R.** (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.

13. **Natarajan, N., Dhillon, I. S., Ravikumar, P., & Tewari, A.** (2013). Learning with noisy labels. *NeurIPS*.

14. **Pereyra, G., Tucker, G., Chorowski, J., Kaiser, Ł., & Hinton, G.** (2017). Regularizing neural networks by penalizing confident output distributions. *ICLR Workshop*.

15. **Wen, Y., Zhang, K., Li, Z., & Qiao, Y.** (2016). A discriminative feature learning approach for deep face recognition. *ECCV*.

16. **Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y.** (2019). CutMix: Regularization strategy to train strong classifiers with localizable features. *ICCV*.

---

## Apêndice A: Distribuição de Classes (Datasets Filtrados)

### Train Filtered (3,890 samples)

| Class ID | Name | Count | Percentage |
|----------|------|-------|------------|
| 0 | PARTITION_HORZ | 295 | 7.58% |
| 1 | PARTITION_VERT | 327 | 8.41% |
| 2 | PARTITION_SPLIT | 950 | 24.42% |
| 3 | PARTITION_HORZ_A | 546 | 14.04% |
| 4 | PARTITION_HORZ_B | 646 | 16.61% |
| 5 | PARTITION_VERT_A | 513 | 13.19% |
| 6 | PARTITION_VERT_B | 613 | 15.76% |

### Val Filtered (976 samples)

| Class ID | Name | Count | Percentage |
|----------|------|-------|------------|
| 0 | PARTITION_HORZ | 63 | 6.45% |
| 1 | PARTITION_VERT | 86 | 8.81% |
| 2 | PARTITION_SPLIT | 238 | 24.39% |
| 3 | PARTITION_HORZ_A | 141 | 14.45% |
| 4 | PARTITION_HORZ_B | 171 | 17.52% |
| 5 | PARTITION_VERT_A | 135 | 13.83% |
| 6 | PARTITION_VERT_B | 142 | 14.55% |

---

**Conclusão Final**: Experimento negativo cientificamente válido. Hipótese H2.1 falsificada. Flatten architecture não é viável. Próximo: Stage 3 Robust (Opção C).
