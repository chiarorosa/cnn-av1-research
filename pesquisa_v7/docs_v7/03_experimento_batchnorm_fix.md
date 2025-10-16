# Experimento 03: BatchNorm Distribution Shift Fix

**Data:** 16/10/2025  
**Solução:** Solution 1 - Conv-Adapter  
**Objetivo:** Resolver BatchNorm distribution shift identificado no doc 01

---

## 1. Motivação

### Problema Identificado (Doc 01, Issue #2)

Na análise crítica da Solução 1 (`01_analise_critica_solucao1.md`), identificamos:

> **Issue #2: BatchNorm in Incorrect Mode**
>
> **Problema:** Durante o treinamento do Stage 2, o backbone congelado mantém BatchNorm layers em **train mode**, causando distribution shift.
>
> **Mecanismo:**
> - `model.train()` coloca TODO o modelo em train mode (incluindo backbone congelado)
> - BatchNorm em train mode → usa estatísticas do batch atual (média/var locais)
> - BatchNorm em eval mode → usa estatísticas globais (running mean/var do Stage 1)
>
> **Consequência:** Features da backbone **variam** entre batches e entre train/val, prejudicando aprendizado dos adapters.

### Fundamentação Teórica

**Ioffe & Szegedy (2015) - Batch Normalization:**
> *"At inference time, we use the population statistics rather than batch statistics to normalize activations."*

**Para modelos congelados:**
- Parâmetros não atualizam (correto) ✅
- Mas BatchNorm statistics **também devem ser congeladas** (estava incorreto) ❌

**He et al. (2016) - Identity Mappings in Deep Residual Networks:**
> *"When fine-tuning pre-trained models, BatchNorm layers should be in eval mode if the corresponding layer's weights are frozen."*

---

## 2. Hipótese

**H1:** BatchNorm em train mode causa instabilidade nas features do backbone, limitando performance dos adapters.

**H2:** Forçar `backbone.eval()` após `model.train()` estabilizará features e permitirá que adapters aprendam modulações mais consistentes.

**Predição:** Val F1: 58.21% (baseline γ=4) → **59-60%** (+1-2 pp)

---

## 3. Implementação

### Fix Aplicado

**Arquivo:** `pesquisa_v7/scripts/020_train_adapter_solution.py`

**Linha 456** (dentro do training loop Stage 2):

```python
# ANTES:
model.train()
train_losses = []
...

# DEPOIS:
model.train()
# CRITICAL FIX (Issue #2): Force backbone BatchNorm to eval mode
# Prevents distribution shift between train/val since backbone is frozen
adapter_backbone.backbone.eval()
train_losses = []
...
```

**Explicação:**
1. `model.train()` ativa dropout e coloca adapters em modo treino ✅
2. `adapter_backbone.backbone.eval()` força BatchNorm do backbone para eval mode ✅
3. Adapters continuam treináveis, mas recebem features **estáveis** do backbone

### Validação da Implementação

**Teste de sanidade:**
```python
# Verificar que BN está realmente em eval mode
import torch
model.train()
adapter_backbone.backbone.eval()

for name, module in adapter_backbone.backbone.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        print(f"{name}: training={module.training}")  # Deve ser False
```

---

## 4. Protocolo Experimental

### Configuração

**Hyperparâmetros** (idênticos ao baseline γ=4):
- Adapter reduction: 4 (baseline)
- Epochs: 50 (early stopping patience=10)
- Batch size: 128
- Learning rate: 0.001 (adapter), 0.0001 (head)
- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Loss: ClassBalancedFocalLoss (gamma=2.0, beta=0.9999)
- Seed: 42

**Mudança experimental:**
- ✅ Adicionar `adapter_backbone.backbone.eval()` após `model.train()`

**Dataset:**
- Train: 363,168 samples (balanceados)
- Validation: 90,793 samples
- Classes: SPLIT (36.37%), RECT (34.11%), AB (29.52%)

### Comando de Execução

```bash
source .venv/bin/activate
cd /home/chiarorosa/CNN_AV1

python3 pesquisa_v7/scripts/020_train_adapter_solution.py \
  --dataset-dir pesquisa_v7/v7_dataset/block_16 \
  --output-dir pesquisa_v7/logs/v7_experiments/solution1_adapter_bn_fix \
  --stage1-checkpoint pesquisa_v7/logs/v7_experiments/solution1_adapter/stage1/stage1_model_best.pt \
  --batch-size 128 \
  --epochs 50 \
  --adapter-reduction 4 \
  --device cuda \
  --seed 42
```

**Baseline para comparação:** `solution1_adapter/stage2_adapter/` (F1=58.21%)

---

## 5. Métricas de Avaliação

### Primárias
1. **Val F1 (macro):** Métrica principal
2. **Train-Val Gap:** Deve diminuir se BN shift estava causando instabilidade
3. **Convergence epoch:** Pode convergir mais rapidamente com features estáveis

### Secundárias
4. **Per-class F1:** Verificar se todas as classes melhoram ou apenas algumas
5. **Loss stability:** Menor variação entre batches
6. **Epoch-to-epoch variance:** Menor oscilação nas métricas

### Critérios de Sucesso

✅ **Sucesso completo:**
- Val F1 ≥ 59.5% (ganho ≥ 1.3 pp)
- Train-Val gap reduzido (mais próximo de 5-8%)
- Convergência estável (sem oscilações)

⚠️ **Sucesso parcial:**
- Val F1 entre 58.5-59.5% (ganho 0.3-1.3 pp)
- Gap ligeiramente melhor
- Convergência similar ou melhor

❌ **Falha:**
- Val F1 < 58.5% (ganho < 0.3 pp ou piora)
- Gap pior ou instabilidade aumentada

---

## 6. Resultados

### Baseline (sem fix)

| Métrica | Valor |
|---------|-------|
| **Val F1** | **58.21%** |
| Train F1 | 57.89% |
| Train-Val Gap | -0.32% (anômalo) |
| Best epoch | 4 |
| Total epochs | 19 |

---

### Experimento (com BatchNorm fix)

| Métrica | Valor | Delta |
|---------|-------|-------|
| **Val F1** | **?%** | **? pp** |
| Train F1 | ?% | ? pp |
| Train-Val Gap | ?% | ? pp |
| Best epoch | ? | ? |
| Total epochs | ? | ? |

*[PREENCHER APÓS EXECUÇÃO]*

---

## 7. Análise Esperada

### Se F1 melhorar (59-60%)

**Confirmação:**
- ✅ BatchNorm distribution shift estava prejudicando performance
- ✅ Features estáveis permitem adapters aprenderem melhor
- ✅ Issue #2 era problema real

**Próximos passos:**
1. Aplicar fix em todos os scripts de treino v7
2. Retreinar Stage 3 (RECT e AB) com fix
3. Documentar como **mandatory practice** para PEFT com backbones congelados

---

### Se F1 não melhorar (< 58.5%)

**Interpretação:**
- BatchNorm shift não era o principal limitante
- Outros problemas têm maior impacto (features, loss, augmentation)

**Próximos passos:**
1. Manter fix (não prejudica, é best practice)
2. Focar em outras melhorias:
   - Loss function ablation (Poly Loss, γ=3.0)
   - Data augmentation (CutMix, MixUp)
   - Stage 1 feature quality

---

## 8. Integração com Tese

### Capítulo 4: Metodologia

**Seção 4.3.3: Implementation Details**

> **BatchNorm Handling for Frozen Backbones**
>
> Durante o treinamento do Stage 2 com backbone congelado, identificamos um problema sutil mas crítico: BatchNorm layers mantinham-se em train mode, causando distribution shift entre batches.
>
> **Solução:** Após `model.train()`, explicitamente forçamos `backbone.eval()` para garantir que BatchNorm use estatísticas globais (frozen do Stage 1), não estatísticas locais por batch.
>
> ```python
> model.train()  # Ativa adapters e dropout
> adapter_backbone.backbone.eval()  # Congela BN statistics
> ```
>
> Este padrão é **essencial** para PEFT com backbones pré-treinados (He et al., 2016; Ioffe & Szegedy, 2015).

### Capítulo 5: Resultados

**Tabela 5.3: Ablation Study - BatchNorm Mode**

| Configuration | Val F1 | Delta | Notes |
|---------------|--------|-------|-------|
| BN train mode (baseline) | 58.21% | - | Distribution shift |
| BN eval mode (fix) | ?% | ? pp | Stable features |

---

## 9. Checklist de Execução

- [x] Fix implementado no script 020
- [x] Protocolo experimental documentado
- [ ] Treinamento executado
- [ ] Métricas coletadas
- [ ] Comparação com baseline
- [ ] Decisão tomada (manter fix ou não)
- [ ] README.md atualizado
- [ ] Integração com tese planejada

---

## 10. Referências

**Ioffe, S., & Szegedy, C. (2015).** *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML.
- Definição de BatchNorm e uso de running statistics em inference

**He, K., et al. (2016).** *Identity Mappings in Deep Residual Networks.* ECCV.
- Best practices para fine-tuning: BN em eval mode para layers congelados

**Chen, H., et al. (2024).** *Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets.* CVPR Workshop.
- Paper base do Conv-Adapter, mas não menciona explicitamente BatchNorm handling

---

**Última atualização:** 16/10/2025 - Criado antes da execução  
**Status:** ⏳ Aguardando execução  
**Próximo passo:** Executar comando e preencher seção 6 (Resultados)
