# Próximos Passos - Pipeline V6 (Atualizado 13/10/2025)

**Última Atualização:** 13/10/2025 18:30  
**Status:** Experimento 09 completo, planejando Experimento 10  
**Responsável:** @chiarorosa

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
