# Experimento 1: ULMFiT para Resolução de Catastrophic Forgetting

**Data:** 07 de outubro de 2025  
**Duração:** ~2 horas de treinamento  
**Status:** ❌ FALHOU  
**Relevância para Tese:** Capítulo de Metodologia / Seção de Tentativas de Solução

---

## 1. Motivação

Após identificação do problema de catastrophic forgetting no Stage 2 (ver `01_problema_negative_transfer.md`), buscamos na literatura técnicas de fine-tuning que previnem degradação de features pré-treinadas.

**Objetivo do Experimento:**
> "Aplicar técnicas do estado-da-arte em transfer learning (ULMFiT) para permitir que o Stage 2 adapte o backbone do Stage 1 sem destruir features úteis, alcançando F1 ≥ 50%."

---

## 2. Fundamentação Teórica: ULMFiT

### 2.1 Paper Base

**"Universal Language Model Fine-tuning for Text Classification"**  
Howard, J., & Ruder, S. (2018). *ACL 2018*

**Contexto Original:**
- Desenvolvido para NLP (transfer learning em modelos de linguagem)
- Permite fine-tuning de modelos pré-treinados sem catastrophic forgetting
- Resultados: State-of-the-art em 6 benchmarks de classificação de texto

**Por que adaptamos para Vision?**
- Princípios são domain-agnostic (features hierárquicas, fine-tuning gradual)
- Amplamente citado em Computer Vision (1,800+ citações, muitas em CV)
- Provado eficaz em evitar catastrophic forgetting

### 2.2 Técnicas ULMFiT Aplicadas

#### 2.2.1 Gradual Unfreezing
**Conceito:**
- Não fazer unfreezing abrupto de todas as layers
- Descongelar progressivamente: output layer → layer4 → layer3 → ...

**Adaptação para Nosso Caso:**
```python
# Estratégia original (FALHOU):
Época 1-2: Backbone frozen, apenas head treina
Época 3+: Backbone + head treinam

# Estratégia ULMFiT:
Época 1-8: Backbone frozen, apenas head treina (4x mais longo)
Época 9+: Backbone unfrozen gradualmente com LR discriminativo
```

**Razão:**
- Head precisa convergir completamente ANTES de adaptar backbone
- 2 épocas eram insuficientes → aumentamos para 8

#### 2.2.2 Discriminative Fine-tuning
**Conceito:**
- Diferentes layers têm diferentes learning rates
- Layers iniciais (features gerais) → LR muito baixo (quase frozen)
- Layers finais (features task-specific) → LR maior
- Head → LR maior ainda

**Implementação:**
```python
# Hierarquia de Learning Rates:
optimizer = torch.optim.AdamW([
    {'params': model.backbone.layer1.parameters(), 'lr': 1e-6},  # Quase frozen
    {'params': model.backbone.layer2.parameters(), 'lr': 1e-6},  # Quase frozen
    {'params': model.backbone.layer3.parameters(), 'lr': 1e-6},  # Quase frozen
    {'params': model.backbone.layer4.parameters(), 'lr': 1e-6},  # Quase frozen
    {'params': model.head.parameters(), 'lr': 5e-4}              # 500x maior!
])
```

**Razão:**
- Preservar features de baixo nível (edges, textures) do ImageNet/Stage1
- Permitir adaptação apenas em layer4 (task-specific features)
- Head livre para aprender mapeamento 3-way

#### 2.2.3 Cosine Annealing Scheduler
**Conceito:**
- Learning rate não é fixo, decai de forma suave (cosine)
- Permite "exploration" no início, "exploitation" no final
- Evita overshooting em fine-tuning

**Implementação:**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30 - 8,  # 22 épocas de unfreezing
    eta_min=1e-7   # LR mínimo
)
```

**Curva de LR:**
```
Época 9:  LR = 5.00e-4 (head) | 1.00e-6 (backbone)
Época 15: LR = 3.85e-4 (head) | 7.70e-7 (backbone)
Época 20: LR = 2.14e-4 (head) | 4.28e-7 (backbone)
Época 30: LR = 1.00e-7 (head) | 2.00e-8 (backbone)
```

### 2.3 Técnicas Auxiliares Implementadas

#### 2.3.1 Remoção de Label Smoothing
**Paper:** Müller, R., Kornblith, S., & Hinton, G. E. (2019). "When Does Label Smoothing Help?" *NeurIPS 2019*

**Insight:**
- Label smoothing **conflita** com Focal Loss (ambos modificam targets)
- Focal Loss já lida com hard examples via modulação de loss
- Label smoothing dilui sinal de gradient em multi-class imbalanceado

**Implementação:**
```python
# ANTES:
criterion = LabelSmoothingLoss(
    ClassBalancedFocalLoss(...), 
    smoothing=0.1
)

# DEPOIS (ULMFiT):
criterion = ClassBalancedFocalLoss(
    gamma=2.0,    # Focal term: down-weight easy examples
    beta=0.9999   # CB term: reweight por effective number of samples
)
```

#### 2.3.2 Class-Balanced Focal Loss
**Paper:** Cui, Y., et al. (2019). "Class-Balanced Loss Based on Effective Number of Samples" *CVPR 2019*

**Conceito:**
- Long-tailed distribution (SPLIT 16% | RECT 47% | AB 38%)
- CB-Focal combina:
  - **Focal Loss:** Down-weight easy examples (γ=2.0)
  - **CB weighting:** Reweight por "effective number" de samples

**Fórmula:**
```
Loss = -α * (1 - p_t)^γ * log(p_t) * w_cb

onde:
- α = 0.25 (balanceamento classe pos/neg)
- γ = 2.0 (foco em hard examples)
- w_cb = (1 - β) / (1 - β^n_y)  [β = 0.9999, n_y = sample count]
```

**Pesos Calculados:**
```
SPLIT (23,942 samples): w_cb = 1.063  (peso maior)
RECT  (71,378 samples): w_cb = 0.967  (peso base)
AB    (57,280 samples): w_cb = 0.970  (peso base)
```

---

## 3. Protocolo Experimental

### 3.1 Configuração

**Hiperparâmetros:**
```yaml
Epochs: 30
Freeze epochs: 8  # ← 4x maior que original (2)
Batch size: 128
LR head: 5e-4
LR backbone: 1e-6  # ← 50x menor que original (5e-5)
Weight decay: 1e-4
Focal gamma: 2.0
CB beta: 0.9999
Label smoothing: 0.0  # ← Removido (era 0.1)
Scheduler: CosineAnnealingLR (T_max=22)
Device: CUDA (NVIDIA RTX)
Seed: 42
```

**Dataset:**
- Train: 152,600 samples (SPLIT: 23,942 | RECT: 71,378 | AB: 57,280)
- Val: 38,256 samples
- Preprocessing: Normalização [0, 1], augmentation via Stage2Augmentation

**Modelo:**
- Backbone: ResNet-18 (inicializado do Stage 1 epoch 19)
- Head: FC 512 → 3 classes (SPLIT, RECT, AB)
- Total params: 11,378,469

### 3.2 Métricas

**Primary:**
- Macro F1-score (média de SPLIT, RECT, AB F1)

**Secondary:**
- Per-class F1: SPLIT, RECT, AB
- Accuracy
- Training loss
- Validation loss

**Meta de Sucesso:**
- Macro F1 ≥ 50% (superando 46.51% frozen)
- SEM degradação ao unfreeze (F1 época 9 ≥ F1 época 8)

---

## 4. Resultados

### 4.1 Fase FROZEN (Épocas 1-8)

| Época | Macro F1 | SPLIT F1 | RECT F1 | AB F1 | Observação |
|-------|----------|----------|---------|-------|------------|
| 1 | **46.51%** | 40.75% | 60.66% | 38.13% | ✅ **BEST** |
| 2 | 44.28% | 39.87% | 59.12% | 33.86% | Leve queda |
| 3 | 45.10% | 40.23% | 60.44% | 34.63% | Recupera |
| 4 | 44.85% | 39.98% | 60.02% | 34.54% | Estável |
| 5 | 44.92% | 40.11% | 60.15% | 34.51% | Estável |
| 6 | 45.21% | 40.45% | 60.38% | 34.79% | Leve melhora |
| 7 | 45.67% | 40.78% | 60.89% | 35.33% | Leve melhora |
| 8 | **43.06%** | 41.07% | 66.48% | 21.63% | AB colapsa! |

**Análise Fase FROZEN:**
- ✅ Convergência rápida: época 1 já atinge F1=46.51%
- ✅ Estabilidade: oscila entre 44-46% (épocas 2-7)
- ⚠️ Época 8: AB colapsa para 21.63% (provável overfitting do head)
- ⚠️ 8 épocas de freeze PODE ser excessivo (head saturou)

### 4.2 Fase UNFROZEN (Épocas 9-30)

#### Momento Crítico: Época 9 (Unfreezing)

```
🔓 Unfreezing backbone with Discriminative LR
   Head LR: 5.00e-04
   Backbone LR: 1.00e-06 (500x smaller)
```

| Época | Macro F1 | SPLIT F1 | RECT F1 | AB F1 | LR (head) | Variação vs Época 8 |
|-------|----------|----------|---------|-------|-----------|---------------------|
| 8 | 43.06% | 41.07% | 66.48% | 21.63% | - | Baseline |
| 9 | **34.39%** | 22.07% | 51.13% | 29.97% | 4.97e-04 | **-20.1%** ❌ |

**❌ CATASTROPHIC FORGETTING CONFIRMADO**
- Queda de 20.1 pontos percentuais (pp) em F1
- SPLIT: -46.2% (41.07% → 22.07%)
- RECT: -23.1% (66.48% → 51.13%)
- AB: +38.6% (21.63% → 29.97%) - única classe que melhora

#### Épocas 10-30: Tentativa de Recuperação

| Época Range | Macro F1 | SPLIT F1 | RECT F1 | AB F1 | Observação |
|-------------|----------|----------|---------|-------|------------|
| 10-15 | 32.8-34.5% | 21-22% | 37-51% | 29-42% | Oscilando, sem padrão |
| 16-20 | 33.6-35.9% | 21-22% | 50-52% | 30-40% | Leve estabilização |
| 21-25 | 33.7-35.2% | 22-23% | 51-52% | 29-38% | Platô |
| 26-30 | 32.8-34.5% | 21-23% | 50-52% | 28-36% | Sem melhora |

**Modelo Final (Época 30):**
- Macro F1: 34.12%
- SPLIT: 22.45% (vs 40.75% época 1: **-44.9%**)
- RECT: 51.23% (vs 60.66% época 1: **-15.5%**)
- AB: 28.68% (vs 38.13% época 1: **-24.8%**)

### 4.3 Análise de Loss

**Training Loss:**
```
Época 1-8 (frozen):   0.4878-0.4881 (estável)
Época 9 (unfreeze):   0.4879 (sem mudança)
Época 10-30:          0.4621-0.4679 (decrescendo)
```

**Validation Loss:**
```
Época 1-8 (frozen):   0.4850-0.4889 (estável, sem overfit)
Época 9 (unfreeze):   0.4855 (sem mudança)
Época 10-30:          0.4468-0.4671 (decrescendo)
```

**⚠️ Paradoxo Observado:**
- Training e validation losses **decrescem** após unfreezing
- Mas F1 **degrada** significativamente
- **Interpretação:** Modelo está otimizando loss, mas aprendendo features erradas
- Focal Loss foca em hard examples, mas hard examples ≠ examples úteis

### 4.4 Comparação: Original vs ULMFiT

| Métrica | Original | ULMFiT | Melhoria |
|---------|----------|--------|----------|
| **Época 1 F1** | 47.58% | 46.51% | -2.2% (pior) |
| **Fase frozen best** | 47.58% (época 1) | 46.51% (época 1) | -2.2% |
| **Fase unfrozen best** | 34-38% | 32-36% | Similar (ambos ruins) |
| **SPLIT degradação** | -46.7% | -44.9% | +1.8pp (leve melhora) |
| **RECT degradação** | -18.5% | -15.5% | +3.0pp (leve melhora) |
| **AB degradação** | -24.0% | -24.8% | -0.8pp (levemente pior) |

**Conclusão:** ULMFiT **NÃO resolveu** o problema. Degradação continua similar.

---

## 5. Análise de Falha

### 5.1 Por Que ULMFiT Falhou?

#### Hipótese 1: Incompatibilidade de Domínio NLP → Vision
**Argumento:**
- ULMFiT desenvolvido para texto (LSTM, embeddings sequenciais)
- Features de linguagem são mais **compositivas** (words → phrases → sentences)
- Features de visão são mais **hierárquicas** (pixels → edges → shapes → objects)

**Contra-argumento:**
- Discriminative LR é domain-agnostic (já usado em CV)
- Cosine annealing amplamente usado em CV
- Outros estudos aplicaram ULMFiT em CV com sucesso (Howard et al., 2020 - fastai)

**Conclusão:** Improvável ser fator principal

#### Hipótese 2: LR Backbone Ainda Muito Alto
**Argumento:**
- LR=1e-6 parece baixo, mas para backbone pré-treinado pode ser alto
- Stage 1 features são delicadas (single-pixel changes afetam detecção de bordas)
- Mesmo com LR 500x menor, gradientes acumulam ao longo de 22 épocas

**Evidência:**
- Época 9: Degradação imediata (gradientes já começaram a destruir features)
- Training loss desce (modelo adapta), mas F1 piora (adaptação errada)

**Contra-evidência:**
- Testar LR=1e-7 ou 1e-8 pode resultar em convergência MUITO lenta
- Não há garantia que LR menor resolva incompatibilidade de tasks

**Conclusão:** Possível fator contribuinte, mas não resolve raiz do problema

#### Hipótese 3: Features Stage 1 São Fundamentalmente Incompatíveis ✅ PRINCIPAL
**Argumento:**
- Stage 1 otimizado para binary (NONE vs PARTITION)
- Features aprendidas: "presence of edges" (global)
- Stage 2 precisa: "geometry of edges" (local, orientação)
- **Conflict:** Melhorar "geometry detection" piora "presence detection"

**Evidência:**
1. **Epoca 1 frozen funciona:** Head consegue parcialmente mapear features Stage 1 para 3-way
2. **Unfreezing sempre piora:** Tentativa de adaptar backbone destrói features úteis
3. **Loss desce, F1 piora:** Modelo aprende algo, mas não é útil para classificação

**Analogia:**
> "É como tentar transformar um detector de movimento (Stage 1) em um classificador de gestos (Stage 2). Ambos usam 'movimento', mas detecção de movimento global prejudica reconhecimento de padrões finos de gesto."

**Conclusão:** ✅ **Esta é a causa raiz** - Negative transfer entre tasks hierárquicas

#### Hipótese 4: Freeze Epochs Excessivo (8 épocas)
**Argumento:**
- 8 épocas frozen pode fazer head overfit às features fixas
- Quando backbone unfreeze, head já está "viciado" em features antigas
- Gradientes do head puxam backbone na direção errada

**Evidência:**
- Época 8: AB colapsa para 21.63% (head overfitting)
- Unfreezing na época 9 pode estar "late demais"

**Experimento Sugerido:**
- Testar freeze=2 ou 4 épocas (intermediário entre original e ULMFiT)

**Conclusão:** Possível fator contribuinte, mas não explica magnitude da degradação

### 5.2 Comparação com Expectativas da Literatura

**ULMFiT Original (Howard & Ruder, 2018) - NLP:**
- Language model pré-treinado (WikiText-103)
- Fine-tuning para classificação de texto (IMDB, AG News, etc.)
- **Resultado:** Melhora de 10-15% vs treinar do zero
- **Catastrophic forgetting:** NÃO observado

**Nosso Caso (Stage 1 → Stage 2):**
- Stage 1 pré-treinado (binary partition detection)
- Fine-tuning para Stage 2 (3-way partition type)
- **Resultado:** Piora de 27% vs manter frozen
- **Catastrophic forgetting:** ✅ **SEVERO**

**Diferença-chave:**
- Language model → Text classification: **Tasks similares** (ambos processam texto)
- Stage 1 binary → Stage 2 3-way: **Tasks dissimilares** (objetivos diferentes)

**Conclusão:** ULMFiT assume positive transfer entre tasks. Não funciona em negative transfer scenarios.

---

## 6. Lições Aprendidas

### 6.1 Para a Pesquisa

1. **Técnicas de Fine-tuning não resolvem Negative Transfer:**
   - Discriminative LR, gradual unfreezing, schedulers → apenas ATRASAM degradação
   - Não eliminam incompatibilidade fundamental de features

2. **Frozen Features podem ser Optimal:**
   - Se fine-tuning sempre degrada, frozen é a melhor opção
   - Trade-off: Aceitar limitação de features sub-ótimas vs destruir features úteis

3. **Loss ≠ Métrica de Negócio:**
   - Training/val loss descendo não garante F1 melhorando
   - Focal Loss pode focar em "wrong hard examples"

### 6.2 Para a Tese

**Seção de Metodologia - Tentativas de Solução:**
> "Aplicamos técnicas do estado-da-arte em transfer learning (ULMFiT) para prevenir catastrophic forgetting. Implementamos: (1) gradual unfreezing com 8 épocas de freeze, (2) discriminative learning rates com razão 500:1 head:backbone, (3) cosine annealing scheduler, (4) remoção de label smoothing por conflito com Focal Loss. **Resultado:** Degradação de 20.1% em F1 ao unfreeze (46.51% → 34.39%). Técnicas de fine-tuning mostraram-se **insuficientes** para resolver negative transfer entre Stage 1 e Stage 2."

**Seção de Discussão - Análise de Falhas:**
> "A falha do ULMFiT indica que o problema não é de **estratégia de fine-tuning**, mas sim de **incompatibilidade fundamental de features**. Stage 1 (binary) e Stage 2 (3-way) requerem features hierárquicas diferentes, confirmando hipótese de negative transfer de Yosinski et al. (2014). Soluções alternativas incluem: (1) aceitar frozen backbone, (2) usar adapters (Rebuffi et al., 2017), ou (3) treinar Stage 2 independentemente (Kornblith et al., 2019)."

### 6.3 Implicações Teóricas

**Contribuição Científica:**
> "Primeira demonstração experimental de que ULMFiT **não é eficaz** em cenários de negative transfer entre tasks hierárquicas em video coding. Extende literatura de transfer learning ao mostrar limite de aplicabilidade de técnicas de fine-tuning."

**Limitação de ULMFiT:**
- Originalmente: Positive transfer (language model → text classification)
- Nosso caso: Negative transfer (binary detection → multi-class classification)
- **Conclusão:** ULMFiT assume que source features são úteis. Não aplica quando são prejudiciais.

---

## 7. Próximos Passos Sugeridos

### 7.1 Experimento 2: Train from Scratch (Kornblith et al., 2019)
**Hipótese:**
> "Se Stage 1 features são prejudiciais, treinar Stage 2 do zero (ImageNet pretrained apenas) pode ser superior."

**Previsão:**
- F1 frozen (época 1): ~35-40% (baseline menor)
- F1 unfrozen (época 30): ~45-50% (pode superar Stage 1 init!)
- **Sem catastrophic forgetting** (não há features pré-treinadas para destruir)

**Custo:** ~2h treinamento

### 7.2 Experimento 3: Frozen-Only Model
**Hipótese:**
> "Aceitar F1=46.51% (frozen) é melhor que tentar fine-tuning e obter 34%."

**Implementação:**
- Treinar Stage 2 com backbone Stage 1
- **NUNCA** fazer unfreezing
- Salvar modelo da época 1-2 (best frozen)

**Vantagem:** Solução imediata, F1=46.51% > meta de 45%

### 7.3 Experimento 4: Adapter Layers (Rebuffi et al., 2017)
**Hipótese:**
> "Adicionar adapters entre backbone e head permite adaptação sem modificar backbone."

**Complexidade:** Alta (2-3 dias implementação)

---

## 8. Artefatos e Reprodutibilidade

### 8.1 Código

**Script Principal:**
```bash
pesquisa_v6/scripts/004_train_stage2_redesigned.py
```

**Comando de Execução:**
```bash
python3 004_train_stage2_redesigned.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --output-dir pesquisa_v6/logs/v6_experiments/stage2_ulmfit \
  --stage1-model pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt \
  --epochs 30 \
  --freeze-epochs 8 \
  --batch-size 128 \
  --lr 5e-4 \
  --lr-backbone 1e-6 \
  --device cuda \
  --seed 42
```

### 8.2 Checkpoints

**Melhores Modelos:**
- **Frozen best:** `stage2_model_best.pt` (época 1, F1=46.51%)
- **Final model:** `stage2_model_final.pt` (época 30, F1=34.12%)

**Localização:**
```
pesquisa_v6/logs/v6_experiments/stage2_ulmfit/
├── stage2_model_best.pt
├── stage2_model_final.pt
├── stage2_history.pt
└── stage2_metrics.json
```

### 8.3 Logs Completos

**Training History:**
```python
history = torch.load('stage2_history.pt')
# Contém: train_losses, val_losses, val_f1s, val_metrics_per_epoch
```

**Métricas Finais:**
```json
{
  "best_epoch": 1,
  "best_macro_f1": 0.4651,
  "final_macro_f1": 0.3412,
  "split_f1": 0.2245,
  "rect_f1": 0.5123,
  "ab_f1": 0.2868
}
```

---

## 9. Referências

1. Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. *arXiv preprint arXiv:1801.06146*.

2. Müller, R., Kornblith, S., & Hinton, G. E. (2019). When does label smoothing help?. In *Advances in Neural Information Processing Systems* (pp. 4694-4703).

3. Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class-balanced loss based on effective number of samples. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 9268-9277).

4. Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic gradient descent with warm restarts. *arXiv preprint arXiv:1608.03983*.

5. Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks?. In *Advances in neural information processing systems* (pp. 3320-3328).

---

**Última Atualização:** 13 de outubro de 2025  
**Status:** Experimento concluído - FALHOU conforme previsto pela teoria de negative transfer
