# Experimento 04: Loss Function Ablation

**Data de Início:** 16/10/2025  
**Pesquisador:** Chiaro Rosa  
**Status:** 🔄 EM PLANEJAMENTO

---

## 1. Motivação

### Contexto do Problema

Após 3 experimentos (baseline, adapter capacity, BatchNorm fix), o **F1 do Stage 2 estagnou em ~58.5%**:

```
Baseline (γ=4):        58.21%
Exp 02 (γ=2):          58.18%  (-0.04 pp, sem ganho)
Exp 03 (BN fix):       58.53%  (+0.32 pp, ganho pequeno)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Progresso total:       +0.32 pp em 3 experimentos
```

**Hipótese atual:** O **gargalo está na loss function**, não na arquitetura.

### Por Que a Loss Function?

#### Problema 1: Hard Negatives Insuficientemente Penalizados

**Loss atual:** ClassBalancedFocalLoss com γ=2.0

```python
FL(pt) = -αt * (1 - pt)^γ * log(pt)
```

Onde:
- `pt` = probabilidade da classe correta
- `γ = 2.0` = focusing parameter (penaliza erros confiantes)
- `αt` = class weight (balanceamento)

**Problema identificado:**
- γ=2.0 é padrão de Lin et al. (2017) para object detection
- Mas **particionamento AV1 pode ser mais difícil** (classes muito similares)
- **Erros confiantes** (high confidence, wrong class) podem não estar sendo punidos suficientemente

#### Problema 2: Cross-Entropy Pode Não Ser Ideal

**Leng et al. (2022)** demonstraram que:
- Cross-entropy saturates gradients para classes hard
- Poly Loss (substituição polinomial) mantém gradients mais ativos
- **Ganhos reportados:** +1.2 pp (ImageNet), +2.3 pp (COCO)

#### Problema 3: Assimetria Entre False Positives e False Negatives

**Ridnik et al. (2021)** mostraram que:
- Multi-label classification se beneficia de penalidades assimétricas
- False Positives e False Negatives têm impactos diferentes
- **Asymmetric Loss** trouxe +0.6 pp (MS-COCO), +1.4 pp (NUS-WIDE)

---

## 2. Revisão de Literatura

### 2.1 Focal Loss (Lin et al., 2017)

**Paper:** "Focal Loss for Dense Object Detection"  
**Venue:** ICCV 2017  
**Citations:** 15,000+

**Contribuição:**
- Introduziu focusing parameter γ para penalizar hard negatives
- γ=0 → Cross-Entropy padrão
- γ=2 → Focal Loss (padrão)
- γ=5 → Penalização extrema

**Resultados:**
- Object detection: +3.9 AP (COCO)
- γ=2 foi ótimo para detection
- γ>3 trouxe instabilidade em alguns datasets

**Aplicação ao nosso caso:**
- Atual: γ=2.0
- **Testaremos γ=3.0:** Maior penalização para hard negatives

---

### 2.2 Poly Loss (Leng et al., 2022)

**Paper:** "PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions"  
**Venue:** NeurIPS 2022  
**Citations:** 200+

**Contribuição:**
- Reformula cross-entropy como série polinomial de Taylor
- Adiciona termos polinomiais de ordem superior
- Mantém gradientes ativos para hard samples

**Formulação:**

```
PolyLoss = CE + ε1 * Poly1(pt) + ε2 * Poly2(pt) + ...
```

Onde:
- `Poly1(pt) = (1 - pt)`
- `Poly2(pt) = (1 - pt)^2`
- `ε1, ε2` = coeficientes (tipicamente ε1=1.0)

**Resultados originais:**
- ImageNet: +1.2 pp (top-1 accuracy)
- COCO detection: +2.3 AP
- Semantic segmentation: +1.8 mIoU

**Por que pode funcionar para AV1:**
- Partições AB são **hard classes** (baixo F1 atual)
- Poly Loss mantém gradientes ativos → melhora aprendizado hard classes

---

### 2.3 Asymmetric Loss (Ridnik et al., 2021)

**Paper:** "Asymmetric Loss For Multi-Label Classification"  
**Venue:** ICCV 2021  
**Citations:** 500+

**Contribuição:**
- Penalidades diferentes para False Positives (FP) e False Negatives (FN)
- `γ_pos` para positivos, `γ_neg` para negativos
- Útil quando FP e FN têm custos diferentes

**Formulação:**

```
ASL = (1 - pt)^γ_pos * log(pt)     [se y=1]
      pt^γ_neg * log(1 - pt)       [se y=0]
```

**Resultados originais:**
- MS-COCO: +0.6 mAP
- NUS-WIDE: +1.4 mAP
- OpenImages: +1.1 mAP

**Aplicação ao nosso caso:**
- AV1: FN (miss SPLIT) pode ser **mais custoso** que FP (predict SPLIT errado)
- Testaremos `γ_pos=2, γ_neg=4` (penalizar mais FN)

---

### 2.4 Label Smoothing (Müller et al., 2019)

**Paper:** "When Does Label Smoothing Help?"  
**Venue:** NeurIPS 2019  
**Citations:** 3,000+

**Contribuição:**
- Suaviza one-hot labels: [0, 1, 0] → [ε, 1-ε, ε]
- Reduz overconfidence
- Melhora calibration (confidence ≈ accuracy)

**Formulação:**

```
y_smooth = (1 - ε) * y_hard + ε / K
```

Onde:
- `ε = 0.1` (padrão)
- `K = num_classes`

**Resultados:**
- ImageNet: +0.2 pp (pequeno ganho, mas melhora calibration)
- CIFAR-100: +0.5 pp

**Por que pode funcionar:**
- Stage 2 tem classes similares (SPLIT, RECT, AB)
- Label smoothing pode evitar overconfidence em fronteiras ambíguas

---

## 3. Hipóteses

### H1: Focal Loss γ=3.0 (Penalização Maior)

**Hipótese:** Aumentar γ de 2.0 para 3.0 melhora F1 das classes hard (AB).

**Fundamentação:**
- Lin et al. (2017): γ controla penalização de hard negatives
- γ=2 é padrão para object detection (task mais simples)
- AV1 partition pode ser mais difícil → requer γ maior

**Predição:** 
- **Val F1:** 58.53% → **60.0-61.0%** (+1.5-2.5 pp)
- **AB F1:** Esperado maior ganho (classe mais hard)

**Risco:** γ muito alto pode causar instabilidade (Lin et al. reportaram issues com γ=5)

---

### H2: Poly Loss (Gradientes Ativos)

**Hipótese:** Poly Loss mantém gradientes ativos para hard samples → melhora AB F1.

**Fundamentação:**
- Leng et al. (2022): +1.2-2.3 pp em ImageNet/COCO
- Cross-entropy satura gradientes para pt próximo de 0 ou 1
- AB classes têm baixo F1 → são hard samples

**Predição:**
- **Val F1:** 58.53% → **60.5-61.5%** (+2.0-3.0 pp)
- **AB F1:** Ganho esperado > SPLIT/RECT

**Configuração:** ε1=1.0 (padrão Leng et al.)

---

### H3: Asymmetric Loss (Penalizar Mais FN)

**Hipótese:** Penalizar mais FN (miss SPLIT) melhora recall → aumenta F1.

**Fundamentação:**
- Ridnik et al. (2021): +0.6-1.4 mAP em multi-label
- AV1: FN (não detectar SPLIT) pode ser mais custoso que FP
- Asymmetric Loss permite ajustar trade-off precision/recall

**Predição:**
- **Val F1:** 58.53% → **59.5-60.5%** (+1.0-2.0 pp)
- **Recall:** Ganho esperado > precision

**Configuração:** γ_pos=2, γ_neg=4 (penalizar mais FN)

---

### H4: Focal Loss + Label Smoothing (Híbrido)

**Hipótese:** Combinar Focal Loss (hard negatives) + Label Smoothing (calibration) → melhor F1 + calibration.

**Fundamentação:**
- Müller et al. (2019): Label smoothing melhora calibration
- Focal Loss: hard negatives
- Combinação pode trazer benefícios complementares

**Predição:**
- **Val F1:** 58.53% → **59.0-60.0%** (+0.5-1.5 pp)
- **Calibration:** Expected Calibration Error (ECE) deve reduzir

**Configuração:** γ=2.0, ε=0.1

---

## 4. Protocolo Experimental

### 4.1 Configuração Base (Fixada para Todas Losses)

**Importante:** Manter **tudo fixo** exceto loss function para ablation limpa.

```python
# Dataset
dataset_dir = "pesquisa_v7/v7_dataset/block_16"
train_split = 80%
val_split = 20%

# Architecture
backbone = ResNet-18 (frozen, pré-treinado Stage 1)
adapter_reduction = 4  (γ=4, 166k params)
adapter_locations = [layer3, layer4]

# Training
batch_size = 128
epochs = 50
early_stopping_patience = 15
optimizer = AdamW
lr_adapter = 0.001
lr_head = 0.0001
weight_decay = 0.01
scheduler = ReduceLROnPlateau (factor=0.5, patience=5)

# Regularization
dropout = [0.1, 0.2, 0.3, 0.4]  (progressive)
BatchNorm handling = backbone.eval()  (fix aplicado)

# Reproducibility
seed = 42
device = cuda
deterministic = True
```

---

### 4.2 Loss Functions a Testar

#### Experimento 4A: Focal Loss γ=3.0

**Objetivo:** Testar se maior penalização de hard negatives melhora F1.

**Mudança:**
```python
# Baseline
criterion = ClassBalancedFocalLoss(gamma=2.0, alpha=class_weights)

# Exp 4A
criterion = ClassBalancedFocalLoss(gamma=3.0, alpha=class_weights)
```

**Output dir:** `pesquisa_v7/logs/v7_experiments/exp04a_focal_gamma3`

---

#### Experimento 4B: Poly Loss

**Objetivo:** Testar se gradientes ativos melhoram hard classes.

**Implementação:**
```python
class PolyLoss(nn.Module):
    """
    Leng et al., NeurIPS 2022
    PolyLoss = CE + ε1 * (1 - pt)
    """
    def __init__(self, epsilon=1.0, class_weights=None):
        super().__init__()
        self.epsilon = epsilon
        self.class_weights = class_weights
        
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, 
                            weight=self.class_weights, 
                            reduction='none')
        pt = torch.exp(-ce)  # pt = exp(-CE)
        poly1 = 1 - pt
        loss = ce + self.epsilon * poly1
        return loss.mean()
```

**Output dir:** `pesquisa_v7/logs/v7_experiments/exp04b_poly_loss`

---

#### Experimento 4C: Asymmetric Loss

**Objetivo:** Testar penalização assimétrica FP vs FN.

**Implementação:**
```python
class AsymmetricLoss(nn.Module):
    """
    Ridnik et al., ICCV 2021
    Adaptado para multi-class (one-vs-rest)
    """
    def __init__(self, gamma_pos=2, gamma_neg=4, class_weights=None):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.class_weights = class_weights
        
    def forward(self, logits, targets):
        # Convert to one-hot
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # Softmax
        probs = F.softmax(logits, dim=1)
        
        # Asymmetric penalties
        pos_loss = targets_one_hot * torch.pow(1 - probs, self.gamma_pos) * torch.log(probs + 1e-8)
        neg_loss = (1 - targets_one_hot) * torch.pow(probs, self.gamma_neg) * torch.log(1 - probs + 1e-8)
        
        loss = -(pos_loss + neg_loss)
        
        # Apply class weights
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = loss.mean(dim=1) * weights
            
        return loss.mean()
```

**Output dir:** `pesquisa_v7/logs/v7_experiments/exp04c_asymmetric_loss`

---

#### Experimento 4D: Focal + Label Smoothing

**Objetivo:** Testar combinação (hard negatives + calibration).

**Implementação:**
```python
class FocalLossWithLabelSmoothing(nn.Module):
    """
    Focal Loss + Label Smoothing
    """
    def __init__(self, gamma=2.0, epsilon=0.1, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.class_weights = class_weights
        
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        
        # Label smoothing
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / num_classes
        
        # Focal loss
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        focal_weight = torch.pow(1 - probs, self.gamma)
        loss = -focal_weight * targets_smooth * log_probs
        
        # Class weights
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = loss.sum(dim=1) * weights
        else:
            loss = loss.sum(dim=1)
            
        return loss.mean()
```

**Output dir:** `pesquisa_v7/logs/v7_experiments/exp04d_focal_label_smoothing`

---

### 4.3 Ordem de Execução

**Executar em paralelo** (4 GPUs ou sequencial):

1. **Exp 4A** (Focal γ=3): ~2h training
2. **Exp 4B** (Poly Loss): ~2h training
3. **Exp 4C** (Asymmetric): ~2h training
4. **Exp 4D** (Focal + LS): ~2h training

**Total:** ~8h se sequencial, ~2h se paralelo

---

## 5. Métricas de Avaliação

### 5.1 Métricas Primárias

**Métrica principal:** Validation F1-score (macro)

**Métricas secundárias:**
- Per-class F1 (SPLIT, RECT, AB)
- Precision (macro)
- Recall (macro)
- Accuracy

**Threshold de sucesso:**
- Ganho > **+1.0 pp** é significativo
- Ganho > **+2.0 pp** é breakthrough

---

### 5.2 Métricas de Calibration

**Para Exp 4D (Label Smoothing):**

- **Expected Calibration Error (ECE):**
  ```
  ECE = Σ (|confidence - accuracy| * bin_size)
  ```
  
- **Maximum Calibration Error (MCE):**
  ```
  MCE = max(|confidence - accuracy|)
  ```

**Threshold:** ECE < 0.10 é bem calibrado

---

### 5.3 Análise de Hard Classes

**Foco especial em AB (classe mais difícil):**

- AB F1 atual: ~10-15% (v5), unknown (v7)
- **Target:** AB F1 > 20%

**Análise:**
- Confusion matrix: onde AB é confundido?
- Confidence distribution: AB tem low confidence?

---

## 6. Análise Planejada

### 6.1 Comparação Quantitativa

**Tabela 1: Overall Performance**

| Loss Function | Val F1 | Delta | Precision | Recall | Accuracy |
|---------------|--------|-------|-----------|--------|----------|
| Baseline (γ=2) | 58.53% | - | - | - | - |
| Exp 4A (γ=3) | ? | ? | ? | ? | ? |
| Exp 4B (Poly) | ? | ? | ? | ? | ? |
| Exp 4C (Asym) | ? | ? | ? | ? | ? |
| Exp 4D (Focal+LS) | ? | ? | ? | ? | ? |

**Tabela 2: Per-Class F1**

| Loss Function | SPLIT F1 | RECT F1 | AB F1 | Mean F1 |
|---------------|----------|---------|-------|---------|
| Baseline | ? | ? | ? | 58.53% |
| Exp 4A | ? | ? | ? | ? |
| Exp 4B | ? | ? | ? | ? |
| Exp 4C | ? | ? | ? | ? |
| Exp 4D | ? | ? | ? | ? |

---

### 6.2 Análise Estatística

**Questões:**
1. Qual loss trouxe maior ganho?
2. Qual loss beneficiou mais AB (hard class)?
3. Há trade-off precision/recall?
4. Label smoothing melhorou calibration?

**Testes:**
- Paired t-test (comparar losses)
- Effect size (Cohen's d)

---

### 6.3 Análise Qualitativa

**Confusion matrices:**
- Onde cada loss erra?
- AB é confundido com RECT ou SPLIT?

**Confidence distributions:**
- Losses produzem confidences diferentes?
- Poly Loss tem confidences mais altas para AB?

---

## 7. Critérios de Sucesso

### 7.1 Sucesso Completo ✅

- **Val F1 > 60.5%** (+2.0 pp sobre baseline 58.53%)
- **AB F1 > 20%** (ganho significativo em hard class)
- Training estável (sem divergência)
- Reprodutível (seed 42)

---

### 7.2 Sucesso Parcial ⚠️

- **Val F1 > 59.5%** (+1.0 pp)
- AB F1 aumentou (mesmo se < 20%)
- Trade-offs aceitáveis (e.g., -1% precision, +3% recall)

---

### 7.3 Falha ❌

- Val F1 < 59.0% (< +0.5 pp)
- AB F1 não melhorou ou piorou
- Training instável (NaN losses, divergência)

---

## 8. Riscos e Mitigação

### Risco 1: Instabilidade com γ=3.0

**Problema:** Lin et al. (2017) reportaram instabilidade com γ > 3

**Mitigação:**
- Monitorar loss/metrics a cada epoch
- Se divergir, testar γ=2.5 (intermediário)
- Reduzir LR se necessário

---

### Risco 2: Poly Loss Requer Tuning

**Problema:** ε1 ideal pode não ser 1.0

**Mitigação:**
- Começar com ε1=1.0 (padrão Leng et al.)
- Se não funcionar, testar ε1=0.5 ou ε1=2.0

---

### Risco 3: Asymmetric Loss para Multi-Class

**Problema:** Original é para multi-label, adaptamos para multi-class

**Mitigação:**
- Implementar one-vs-rest approach
- Validar implementação com toy example
- Comparar com literatura (buscar adaptações existentes)

---

### Risco 4: Nenhuma Loss Melhora

**Problema:** Loss function não é o gargalo

**Mitigação:**
- Documentar resultado negativo (PhD-level)
- Prosseguir para Data Augmentation (próxima prioridade)
- Considerar que problema está em Stage 1 features

---

## 9. Cronograma

### Fase 1: Implementação (0.5h)

- [ ] Criar `v7_pipeline/losses_ablation.py`
- [ ] Implementar PolyLoss, AsymmetricLoss, FocalLabelSmoothing
- [ ] Validar implementações com toy examples
- [ ] Unit tests (verificar gradientes)

### Fase 2: Treinamento (8h sequencial ou 2h paralelo)

- [ ] Exp 4A: Focal γ=3.0 (~2h)
- [ ] Exp 4B: Poly Loss (~2h)
- [ ] Exp 4C: Asymmetric Loss (~2h)
- [ ] Exp 4D: Focal + Label Smoothing (~2h)

### Fase 3: Análise (1h)

- [ ] Extrair métricas de todos experiments
- [ ] Gerar tabelas comparativas
- [ ] Confusion matrices
- [ ] Confidence distributions
- [ ] Testes estatísticos

### Fase 4: Documentação (1h)

- [ ] `04b_resultados_loss_ablation.md`
- [ ] Integrar com tese (Capítulo 5)
- [ ] Decidir próximo passo

**Total estimado:** ~10h

---

## 10. Artefatos

### Código
```
pesquisa_v7/v7_pipeline/losses_ablation.py  (novas losses)
pesquisa_v7/scripts/021_train_loss_ablation.py  (script de treino)
```

### Checkpoints
```
pesquisa_v7/logs/v7_experiments/exp04a_focal_gamma3/
pesquisa_v7/logs/v7_experiments/exp04b_poly_loss/
pesquisa_v7/logs/v7_experiments/exp04c_asymmetric_loss/
pesquisa_v7/logs/v7_experiments/exp04d_focal_label_smoothing/
```

### Documentação
```
pesquisa_v7/docs_v7/04_experimento_loss_function_ablation.md  (este doc)
pesquisa_v7/docs_v7/04b_resultados_loss_ablation.md  (resultados)
```

---

## 11. Referências

1. **Lin et al. (2017)** - "Focal Loss for Dense Object Detection", ICCV 2017
2. **Leng et al. (2022)** - "PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions", NeurIPS 2022
3. **Ridnik et al. (2021)** - "Asymmetric Loss For Multi-Label Classification", ICCV 2021
4. **Müller et al. (2019)** - "When Does Label Smoothing Help?", NeurIPS 2019
5. **Cui et al. (2019)** - "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019

---

## 12. Checklist de Execução

### Antes de Começar
- [ ] Confirmar baseline F1=58.53% (Exp 03 BN fix)
- [ ] Dataset pronto (`v7_dataset/block_16/`)
- [ ] Stage 1 checkpoint disponível
- [ ] GPU disponível (8GB VRAM mínimo)

### Durante Experimentos
- [ ] Monitorar loss/metrics a cada epoch
- [ ] Verificar stability (NaN, divergência)
- [ ] Salvar checkpoints (best + final)
- [ ] Log completo de treinamento

### Após Treinamento
- [ ] Extrair métricas de todos experimentos
- [ ] Comparação estatística (paired t-test)
- [ ] Análise qualitativa (confusion matrices)
- [ ] Documentar resultados (04b_resultados_loss_ablation.md)
- [ ] Decidir melhor loss function
- [ ] Atualizar código base se necessário

---

**Última atualização:** 16/10/2025 - 00:30  
**Status:** 🔄 PRONTO PARA IMPLEMENTAÇÃO  
**Próximo passo:** Criar `losses_ablation.py` e script de treino
