# Próximos Passos - Pipeline V6 (Atualizado 13/10/2025 - 20:00)

**Última Atualização:** 13/10/2025 20:00  
**Status:** 🚨 **PROBLEMA CRÍTICO IDENTIFICADO** - Stage 2 colapsado, Experimento 10 bloqueado  
**Responsável:** @chiarorosa

---

## 🚨 PROBLEMA CRÍTICO: Stage 2 Model Colapsado

### Descoberta (13/10/2025 19:30)

Durante a implementação do Script 009 (análise de confusion matrix), **descobriu-se que o modelo Stage 2 está completamente colapsado**:

**Modelo Best (stage2_model_best.pt):**
- Prediz **RECT (classe 1) para 100%** das amostras
- Accuracy: 46.44% (= prevalência de RECT no dataset)
- F1 macro: 0.21 (SPLIT=0.0, RECT=0.63, AB=0.0)

**Modelo Final (stage2_model_final.pt):**
- Prediz **SPLIT (classe 0) para 99.99%** das amostras
- Accuracy: 15.58%
- F1 macro: 0.09

**Análise do History:**
- Época 0 (frozen backbone): F1=0.4651 ✅ Melhor performance
- Épocas 1-7: Estável ~0.44-0.46
- **Época 8:** **COLAPSO** - F1=0.3439, Acc=0.3868 ❌
- Épocas 9-29: Nunca recuperou (~0.33-0.37)

### Causa Identificada

**Catastrophic Forgetting Severo** ao unfreeze do backbone (época 7→8):
- Backbone Stage 1 features incompatíveis com Stage 2 task
- Unfreezing destruiu features aprendidas durante frozen training
- Consistente com documentado em `docs_v6/01_problema_negative_transfer.md`

### Impacto

🚫 **BLOQUEIA Experimento 10 (Confusion-Based Noise Injection)**
- Exp 10 requer confusion matrix realista do Stage 2
- Com modelo colapsado, matriz é trivial (tudo prediz uma classe)
- Noise injection seria inútil

⚠️ **INVALIDA Experimento 09?**
- Se Stage 2 já estava colapsado durante pipeline evaluation:
  * Stage 3 recebeu inputs não-diversificados (só RECT ou só SPLIT)
  * Resultados (45.86% accuracy) podem estar incorretos
  * **NECESSÁRIO:** Re-avaliar pipeline com Stage 2 funcional

---

## 🎯 PLANO DE AÇÃO REVISADO (13-18/10/2025)

### Opção A: Usar Modelo Frozen (RECOMENDADO - 1 dia) ⭐

**Fundamentação:**
- Época 0 (frozen) tinha F1=0.4651 ✅ Melhor que unfrozen
- Script 004 foi projetado para salvar checkpoint frozen
- Unfreezing causou colapso → **não usar unfrozen**

**Protocolo:**
1. **Localizar checkpoint frozen (época 0)** - 10 min
   ```bash
   # Verificar se existe stage2_model_block16_ep0.pt
   ls -lh pesquisa_v6/logs/v6_experiments/stage2/
   ```

2. **Se NÃO existe, retreinar apenas época 0** - 30 min
   ```bash
   python3 pesquisa_v6/scripts/004_train_stage2_redesigned.py \
     --dataset-dir pesquisa_v6/v6_dataset/block_16 \
     --epochs 1 \
     --batch-size 128 \
     --output-dir pesquisa_v6/logs/v6_experiments/stage2_frozen \
     --device cuda \
     --save-every-epoch  # NOVO argumento
   ```

3. **Validar modelo frozen** - 15 min
   ```bash
   python3 pesquisa_v6/scripts/009_analyze_stage2_confusion.py \
     --stage2-model pesquisa_v6/logs/v6_experiments/stage2_frozen/stage2_model_ep0.pt \
     --dataset-dir pesquisa_v6/v6_dataset/block_16 \
     --device cuda
   ```
   
   **Esperado:** F1 ~0.46-0.47, accuracy ~48-49%

4. **Re-avaliar Pipeline Experimento 09** - 30 min
   ```bash
   python3 pesquisa_v6/scripts/008_run_pipeline_eval_v6.py \
     --stage1-model pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt \
     --stage2-model pesquisa_v6/logs/v6_experiments/stage2_frozen/stage2_model_ep0.pt \
     --stage3-rect-model pesquisa_v6/logs/.../stage3_rect_robust.pt \
     --stage3-ab-models <ensemble> \
     --output-dir pesquisa_v6/logs/v6_experiments/pipeline_eval_frozen_s2
   ```
   
   **Verificar:** Accuracy ≈ 45.86% (Exp09) ou melhorou?

5. **Prosseguir Experimento 10** - 3 dias
   - Usar confusion matrix do modelo frozen
   - Implementar confusion-based noise injection
   - Retreinar Stage 3 com noise realista

**Cronograma:**
- **14/10 (Segunda):** Retreinar Stage 2 frozen + validar + re-avaliar pipeline
- **15/10 (Terça):** Implementar confusion-based labels
- **16-17/10 (Quarta-Quinta):** Retreinar Stage 3 com noise
- **18/10 (Sexta):** Avaliação final e decisão

**Ganho esperado:** +1.5-2.5pp (45.86% → 47.5-48.5%)

---

### Opção B: Retreinar Stage 2 Completo (2-3 dias)

**Fundamentação:**
- Implementar salvamento de checkpoints a cada época
- Monitorar F1 por classe durante treinamento
- Adicionar early stopping baseado em F1 macro (não loss)

**Modificações no Script 004:**
```python
# 1. Salvar checkpoint a cada época
if (epoch + 1) % 1 == 0:
    save_checkpoint(
        model, optimizer, epoch,
        f'stage2_model_ep{epoch}.pt'
    )

# 2. Early stopping baseado em F1
best_f1 = 0
patience = 10
patience_counter = 0

for epoch in range(epochs):
    val_f1 = validate_epoch(...)
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        save_checkpoint(model, 'best')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# 3. Monitorar F1 por classe
for i, class_name in enumerate(['SPLIT', 'RECT', 'AB']):
    f1_class = f1_score(y_true, y_pred, labels=[i], average='macro')
    print(f"  {class_name}: F1={f1_class:.4f}")
```

**Cronograma:**
- **14/10:** Modificar Script 004 com melhorias (2h)
- **14/10:** Retreinar Stage 2 com monitoring (2h)
- **15/10:** Validar melhor checkpoint (30min)
- **15/10:** Análise confusion matrix (30min)
- **16-17/10:** Experimento 10 (Confusion-Based Noise)
- **18/10:** Avaliação e decisão

**Risco:** Pode não resolver catastrophic forgetting (é problema arquitetural)

---

### Opção C: Pular Exp 10 → Ir Direto para Train-with-Predictions (3 dias)

**Fundamentação:**
- Confusion-based noise é aproximação de train-with-predictions
- Se vamos usar predições reais, melhor usar direto (não confusion matrix)
- Heigold et al. (2016): Treinar com predições do modelo upstream

**Protocolo:**
```python
# 1. Gerar predições Stage 2 em tempo real
class TrainWithPredictionsDataset(Dataset):
    def __init__(self, stage2_model, stage3_dataset):
        self.stage2_model = stage2_model
        self.stage3_dataset = stage3_dataset
    
    def __getitem__(self, idx):
        block, gt_label, qp = self.stage3_dataset[idx]
        
        # Computar predição Stage 2 (frozen, sem grad)
        with torch.no_grad():
            stage2_logits = self.stage2_model(block.unsqueeze(0))
            stage2_pred = torch.argmax(stage2_logits, dim=1).item()
        
        # 75% usa GT, 25% usa predição Stage 2
        if random.random() < 0.75:
            return block, gt_label, qp  # Clean
        else:
            # Mapear predição Stage 2 → Stage 3 labels
            mapped_label = map_stage2_to_stage3(stage2_pred, head='RECT')
            return block, mapped_label, qp  # Noisy (real prediction)
```

**Vantagens:**
- Noise é **exatamente** o que Stage 3 verá em pipeline real
- Não depende de confusion matrix (robusta a Stage 2 colapsado)
- Mais próximo de "online learning" (adapta a predições reais)

**Desvantagens:**
- Mais complexo de implementar
- Custo computacional maior (forward pass Stage 2 durante treinamento)

**Cronograma:**
- **14/10:** Implementar TrainWithPredictionsDataset (4h)
- **15/10:** Validar implementação (2h)
- **16-17/10:** Retreinar Stage 3 RECT e AB (1.5d)
- **18/10:** Avaliação e decisão

**Ganho esperado:** +1.0-2.0pp (experimental, sem baseline na literatura para video codecs)

---

## 🎯 DECISÃO RECOMENDADA

### Primeira Prioridade: Opção A (Usar Modelo Frozen) ⭐

**Razões:**
1. ✅ **Mais rápido** (1 dia vs 2-3 dias)
2. ✅ **Menor risco** (modelo frozen já provou F1=0.4651)
3. ✅ **Validação científica** (confirma hipótese de que frozen > unfrozen)
4. ✅ **Mantém cronograma** (Exp 10 inicia 15/10)

**Se Opção A falhar** (modelo frozen não disponível e retreino frozen também colapsa):
→ Tentar **Opção C** (Train-with-Predictions)
→ Pular Opção B (retreinar completo é alto custo, baixo ganho esperado)

---

## 📝 Próxima Ação Imediata (HOJE - 13/10/2025 à noite)

### 1. Verificar Existência de Checkpoint Frozen 🔴 URGENTE

```bash
cd /home/chiarorosa/CNN_AV1
ls -lh pesquisa_v6/logs/v6_experiments/stage2/ | grep -E "ep[0-9]|frozen"
```

**Se encontrar `stage2_model_ep0.pt` ou similar:**
✅ Executar Script 009 nele e validar F1 ~0.46

**Se NÃO encontrar:**
⚠️ Retreinar 1 época frozen amanhã (14/10 manhã, 30 min)

### 2. Atualizar Documentação 🟡

- ✅ PROBLEMA_CRITICO_STAGE2.md criado
- ✅ PROXIMOS_PASSOS.md atualizado
- ⏳ Criar `docs_v6/10_stage2_collapse_analysis.md` (PhD-level)

### 3. Push das Mudanças 🟡

```bash
git add pesquisa_v6/PROXIMOS_PASSOS.md
git commit -m "docs: Atualizar PROXIMOS_PASSOS com Opção A/B/C pós-diagnóstico Stage 2"
git push origin main
```

---

## 🗓️ Cronograma Revisado (14-20/10/2025)

| Data | Atividade | Duração | Status |
|------|-----------|---------|--------|
| **13/10 (Domingo)** | Script 009 + Diagnóstico Stage 2 | 4h | ✅ COMPLETO |
| **14/10 (Segunda) AM** | Retreinar Stage 2 frozen (1 época) | 30min | 🔴 PRÓXIMO |
| **14/10 (Segunda) PM** | Validar frozen + Re-avaliar pipeline | 1h | 🔴 PRÓXIMO |
| **15/10 (Terça)** | Implementar confusion-based labels | 1d | ⏳ |
| **16-17/10 (Qua-Qui)** | Retreinar Stage 3 com confusion noise | 1.5d | ⏳ |
| **18/10 (Sexta)** | Avaliação Exp 10 + Checkpoint decisão | 0.5d | ⏳ |
| **19-20/10 (Sáb-Dom)** | Buffer para ajustes / Exp 10.1-10.2 | 2d | ⏳ |

**Meta Final:** Accuracy ≥48.0% até 20/10/2025

---

## 📖 ARQUIVADO: Plano Original Experimento 10 (Pré-Descoberta)

<details>
<summary>Clique para expandir cronograma original (mantido para registro histórico)</summary>

## 🎯 Experimento 10: Confusion-Based Noise Injection

---

## 📊 Estado Atual do Projeto

### Experimentos Completos

| # | Experimento | Status | Resultado | Doc |
|---|-------------|--------|-----------|-----|
| 01-08 | Pipeline V6 Baseline | ✅ | Accuracy=47.66% (-0.34pp da meta) | `05_avaliacao_pipeline_completo.md` |
| **09** | **Noise Injection (Random)** | ✅ | Accuracy=45.86% (-1.80pp), Cascade error -28pp | `09_noise_injection_stage3.md` |

### Descobertas Científicas (Experimento 09)

1. ✅ **Hipótese H3.1 VALIDADA:** Distribution Shift é causa raiz do erro cascata
2. ✅ **Noise Injection funciona:** Reduziu cascade error em 28-56pp
3. ✅ **2 classes recuperadas:** HORZ 0%→23.94%, VERT_A 0%→15.25%
4. ⚠️ **Trade-off confirmado:** Robustez ↑, Accuracy ↓
5. ✅ **Bug crítico descoberto:** IndexError em multi-dataset sampling

### Problema Remanescente

> "Noise injection com **labels aleatórios** melhora robustez mas degrada accuracy. Precisamos de noise mais realista baseado em **erros reais do Stage 2**."

**Gap atual:** Accuracy 45.86% vs meta 48.0% = **-2.14pp**

---

## 🎯 Experimento 10: Confusion-Based Noise Injection

### Objetivo

Substituir labels aleatórios por distribuição de confusão real do Stage 2 para melhorar trade-off robustez/accuracy.

**Meta:** Accuracy ≥48.0% mantendo ganhos de robustez

### Fundamentação Teórica

- **Heigold et al. (2016):** Train-with-predictions reduz cascade error (usado em Google Translate)
- **Hendrycks et al. (2019):** Noise realista > noise uniforme
- **Natarajan et al. (2013):** Distribuição de noise importa para robustez

### Hipótese H3.2

> "Treinar Stage 3 com noise baseado em **confusão real** do Stage 2 (não aleatório) melhorará accuracy mantendo robustez."

**Evidência:** 
- Exp09 provou que noise injection funciona (+28pp cascade error)
- Mas labels aleatórios causaram trade-off negativo (-1.80pp accuracy)
- Stage 2 tem padrões específicos de erro (RECT↔AB confusão esperada ~30-35%)

---

## 📅 Cronograma Experimento 10

### Fase 1: Análise de Confusão (0.5 dia - 14/10)

**Script 009:** `pesquisa_v6/scripts/009_analyze_stage2_confusion.py`

**Objetivo:** Computar matriz de confusão real do Stage 2

**Protocolo:**
```python
# 1. Carregar Stage 2 frozen model (epoch 8)
model = torch.load('logs/v6_experiments/stage2/stage2_model_best.pt')

# 2. Inferir validation set completo (38,256 samples)
predictions, ground_truths = [], []
for batch in val_dataloader:
    preds = model(batch)
    predictions.append(preds)
    ground_truths.append(batch.labels)

# 3. Computar confusion matrix RECT vs AB
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ground_truths, predictions)

# 4. Extrair probabilidades de erro
confusion_probs = {
    "RECT": {
        "AB": cm[RECT, AB] / cm[RECT].sum(),
        "SPLIT": cm[RECT, SPLIT] / cm[RECT].sum(),
        "correct": cm[RECT, RECT] / cm[RECT].sum()
    },
    "AB": {
        "RECT": cm[AB, RECT] / cm[AB].sum(),
        "SPLIT": cm[AB, SPLIT] / cm[AB].sum(),
        "correct": cm[AB, AB] / cm[AB].sum()
    }
}

# 5. Salvar como JSON
import json
with open('confusion_matrix_stage2.json', 'w') as f:
    json.dump(confusion_probs, f, indent=2)
```

**Output esperado:**
```json
{
  "RECT": {
    "AB": 0.35,
    "SPLIT": 0.12,
    "correct": 0.53
  },
  "AB": {
    "RECT": 0.28,
    "SPLIT": 0.15,
    "correct": 0.57
  },
  "SPLIT": {
    "RECT": 0.20,
    "AB": 0.18,
    "correct": 0.62
  }
}
```

**Tempo:** 4 horas (2h impl + 1h exec + 1h análise)

---

### Fase 2: Implementação Confusion-Based (1 dia - 15/10)

**Modificar Scripts 005 e 006:**

```python
class NoisyDataset(Dataset):
    def __init__(self, clean_dataset, noise_datasets, noise_ratio=0.25, 
                 confusion_matrix=None):  # ← NOVO
        self.confusion_matrix = confusion_matrix
        # ... resto igual ...
    
    def __getitem__(self, idx):
        if idx < self.clean_size:
            return self.clean_dataset[self.clean_indices[idx]]
        else:
            noise_idx = idx - self.clean_size
            dataset_idx = noise_idx % len(self.noise_datasets)
            
            local_idx = noise_idx // len(self.noise_datasets)
            local_idx = local_idx % len(self.noise_datasets[dataset_idx])
            sample_idx = self.noise_datasets[dataset_idx][local_idx]
            
            block, _, qp = sample_idx
            
            # ❌ ANTES (labels aleatórios):
            # random_label = torch.randint(0, 2, (1,)).item()
            
            # ✅ DEPOIS (confusion-based):
            noise_source = self.noise_sources[dataset_idx]  # "AB" ou "SPLIT"
            confusion_probs = self.confusion_matrix[noise_source]
            
            # Exemplo: noise_source="AB", confusion_probs={"HORZ": 0.6, "VERT": 0.4}
            labels = list(confusion_probs.keys())
            probs = list(confusion_probs.values())
            random_label = np.random.choice(len(labels), p=probs)
            
            return block, random_label, qp
```

**Argumentos CLI:**
```python
parser.add_argument('--confusion-matrix', type=str, 
                    help='Path to confusion matrix JSON')
```

**Tempo:** 6 horas (4h impl + 2h test)

---

### Fase 3: Retreino com Confusion-Based Noise (1.5 dia - 16-17/10)

**Stage 3-RECT Robust v2:**
```bash
python3 pesquisa_v6/scripts/005_train_stage3_rect.py \
  --dataset-dir pesquisa_v6/v6_dataset_stage3/RECT/block_16 \
  --noise-injection 0.25 \
  --noise-sources AB SPLIT \
  --confusion-matrix pesquisa_v6/logs/v6_experiments/confusion_matrix_stage2.json \
  --epochs 30 \
  --batch-size 128 \
  --output-dir pesquisa_v6/logs/v6_experiments/stage3_rect_robust_v2 \
  --device cuda \
  --stage2-model pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt
```

**Stage 3-AB Robust v2:**
```bash
python3 pesquisa_v6/scripts/006_train_stage3_ab_fgvc.py \
  --dataset_dir pesquisa_v6/v6_dataset_stage3/AB/block_16 \
  --noise-injection 0.25 \
  --noise-sources RECT SPLIT \
  --confusion-matrix pesquisa_v6/logs/v6_experiments/confusion_matrix_stage2.json \
  --phase1_epochs 5 \
  --phase2_epochs 25 \
  --batch_size 128 \
  --output_dir pesquisa_v6/logs/v6_experiments/stage3_ab_robust_v2 \
  --stage2_model pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt
```

**Tempo:** 12 horas (6h RECT + 6h AB)

---

### Fase 4: Avaliação e Análise (0.5 dia - 18/10)

**Pipeline Evaluation:**
```bash
python3 pesquisa_v6/scripts/008_run_pipeline_eval_v6.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --stage1-model pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt \
  --stage2-model pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt \
  --stage3-rect-model pesquisa_v6/logs/v6_experiments/stage3_rect_robust_v2/stage3_rect_model_best.pt \
  --stage3-ab-models \
    pesquisa_v6/logs/v6_experiments/stage3_ab_robust_v2/stage3_ab_fgvc_best.pt \
    pesquisa_v6/logs/v6_experiments/stage3_ab_robust_v2/stage3_ab_fgvc_best.pt \
    pesquisa_v6/logs/v6_experiments/stage3_ab_robust_v2/stage3_ab_fgvc_best.pt \
  --stage1-threshold 0.45 \
  --output-dir pesquisa_v6/logs/v6_experiments/pipeline_eval_robust_v2 \
  --device cuda \
  --batch-size 256
```

**Análise Comparativa:**
```python
# Comparar:
# - Baseline: 47.66%
# - Noise Random (Exp09): 45.86%
# - Noise Confusion (Exp10): ???

# Métricas-chave:
# 1. Overall Accuracy (meta: ≥48.0%)
# 2. HORZ F1 (Exp09: 23.94%)
# 3. VERT_A F1 (Exp09: 15.25%)
# 4. Cascade error RECT/AB
# 5. Stage 3 standalone performance
```

**Tempo:** 4 horas (1h eval + 3h análise + doc)

---

## 🎯 Resultados Esperados (Exp10)

### Cenário Otimista (70% probabilidade)

| Métrica | Baseline | Noise Random (Exp09) | **Noise Confusion (Exp10)** | Meta |
|---------|----------|----------------------|-----------------------------|------|
| **Accuracy** | 47.66% | 45.86% | **48.5-49.2%** ✅ | ≥48.0% |
| HORZ F1 | 0.00% | 23.94% | **28-35%** ✅ | >20% |
| VERT_A F1 | 0.00% | 15.25% | **20-28%** ✅ | >15% |
| Cascade RECT | -93% | -65% | **-55% a -45%** ✅ | <-70% |
| Cascade AB | -94% | -38% | **-30% a -20%** ✅ | <-50% |
| Stage 3-AB standalone | 24.50% | 19.41% | **22-25%** ✅ | >20% |

**Ganho total:** +2.5-3.5pp vs Baseline, +2.6-3.3pp vs Exp09

### Cenário Realista (25% probabilidade)

| Métrica | Valor | Status |
|---------|-------|--------|
| Accuracy | 47.5-48.2% | ⚠️ Próximo da meta |
| HORZ F1 | 25-28% | ✅ Melhorou |
| VERT_A F1 | 18-22% | ✅ Melhorou |

**Ação:** Testar grid search noise ratio (10-35%) ou ensemble real

### Cenário Pessimista (5% probabilidade)

| Métrica | Valor | Status |
|---------|-------|--------|
| Accuracy | <47.5% | ❌ Não melhorou |

**Ação:** Reavaliar estratégia, considerar Train-with-Predictions (Exp11)

---

## 🚀 Experimentos Alternativos (Se Exp10 < 48%)

### Exp 10.1: Grid Search Noise Ratio (1 dia)

**Hipótese:** 25% noise pode não ser ótimo

**Protocolo:**
```bash
for ratio in 0.10 0.15 0.20 0.25 0.30 0.35; do
  python3 005_train_stage3_rect.py \
    --noise-injection $ratio \
    --confusion-matrix confusion_matrix.json \
    --output-dir stage3_rect_noise${ratio}
done
```

**Ganho esperado:** +0.5-1.0pp

### Exp 10.2: Ensemble Real AB (1.5 dia)

**Problema:** Usado 1 modelo repetido 3x

**Solução:**
```bash
for seed in 42 123 456; do
  python3 006_train_stage3_ab_fgvc.py \
    --seed $seed \
    --confusion-matrix confusion_matrix.json \
    --output-dir stage3_ab_seed${seed}
done
```

**Ganho esperado:** +0.3-0.8pp

### Exp 11: Train-with-Predictions (3 dias)

**Conceito:** Substituir noise sintético por predições reais do Stage 2

**Protocolo:**
```python
# Fase 1: Gerar predições Stage 2 (2h)
python3 generate_stage2_predictions.py \
  --stage2-model logs/.../stage2_model_best.pt \
  --dataset v6_dataset/block_16/train.pt \
  --output stage2_predictions_train.pt

# Fase 2: Treinar Stage 3 com predições (não GT)
class PredictionAwareDataset(Dataset):
    def __init__(self, blocks, stage2_predictions):
        self.blocks = blocks
        self.labels = stage2_predictions  # ← Usa predições, não GT
```

**Ganho esperado:** +1.0-2.0pp

**Custo:** 3 dias

---

## 📊 Priorização Final (14-20/10/2025)

| Exp | Técnica | Custo | Ganho | Prioridade | Data |
|-----|---------|-------|-------|------------|------|
| **10** | **Confusion-Based Noise** | **3d** | **+1.5-2.5pp** | 🔴🔴🔴 **PRÓXIMO** | **14-17/10** |
| 10.1 | Grid Search Noise | 1d | +0.5-1.0pp | 🔴🔴 Alta | 18/10 (se <48%) |
| 10.2 | Ensemble Real AB | 1.5d | +0.3-0.8pp | 🔴🔴 Alta | 19-20/10 (se <48%) |
| 11 | Train-with-Predictions | 3d | +1.0-2.0pp | 🔴 Média | 21-23/10 (se <47.5%) |

---

## ✅ Checkpoint de Decisão (18/10/2025)

**Após Exp10 completar:**

### Se Accuracy ≥48.0% → ✅ FINALIZAR Pipeline v6

**Ações:**
1. Documentar Exp10 em `docs_v6/10_confusion_based_noise.md`
2. Atualizar `PLANO_v6_Out.md` com resultados finais
3. Criar apresentação de resultados (slides)
4. Preparar artigo científico (draft)
5. Avançar para encoding speedup experiments

**Tempo:** 3-5 dias documentação

### Se 47.5% ≤ Accuracy < 48.0% → ⚠️ OTIMIZAÇÕES RÁPIDAS

**Ações:**
1. Exp 10.1: Grid search noise ratio (1 dia)
2. Exp 10.2: Ensemble real AB (1.5 dia)
3. Re-avaliar pipeline

**Tempo adicional:** 2.5 dias

### Se Accuracy < 47.5% → ❌ REAVALIAR ESTRATÉGIA

**Ações:**
1. Analisar por que Exp10 falhou
2. Considerar Exp11 (Train-with-Predictions) - 3 dias
3. Ou aceitar 47.5% como limite técnico e documentar
4. Ou explorar arquiteturas alternativas (ViT, etc)

**Decisão crítica:** Investir mais tempo ou finalizar?

---

## 📝 Próxima Ação Imediata (AMANHÃ - 14/10/2025)

### Manhã (4h): Implementar Script 009

```bash
cd /home/chiarorosa/CNN_AV1/pesquisa_v6/scripts

# Criar script
# Conteúdo: Análise de confusão Stage 2
# - Carregar modelo frozen
# - Inferir validation set
# - Computar confusion matrix
# - Salvar JSON

# Testar com 100 samples primeiro
python3 009_analyze_stage2_confusion.py --test
```

### Tarde (2h): Executar análise completa

```bash
# Executar com dataset completo
python3 009_analyze_stage2_confusion.py \
  --stage2-model ../logs/v6_experiments/stage2/stage2_model_best.pt \
  --dataset ../v6_dataset/block_16/val.pt \
  --output ../logs/v6_experiments/confusion_matrix_stage2.json

# Validar output
python3 -c "
import json
with open('confusion_matrix_stage2.json') as f:
    cm = json.load(f)
    print('Confusion Matrix:')
    print(json.dumps(cm, indent=2))
    
    # Verificar somas
    for cls in cm:
        total = sum(cm[cls].values())
        print(f'{cls}: {total:.3f} (deve ser ~1.0)')
"
```

### Análise (2h): Documentar padrões

**Criar:** `docs_v6/10_confusion_analysis.md` (preliminar)

**Conteúdo:**
- Matriz de confusão visualizada
- Principais confusões identificadas
- Hipóteses sobre causas (padrões visuais)
- Expectativa para Exp10

---

**Resumo:** Começar Exp10 amanhã (14/10), completar em 3 dias, checkpoint de decisão em 18/10.

**Meta final:** Accuracy ≥48.0% até 20/10/2025.
