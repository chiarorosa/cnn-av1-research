# PROBLEMA CRÍTICO IDENTIFICADO: Stage 2 Model Colapsado

**Data:** 13/10/2025 19:30  
**Responsável:** Chiaro Rosa  
**Status:** 🚨 CRÍTICO - BLOQUEADOR do Experimento 10

---

## 🚨 Descoberta

Durante a implementação do Script 009 (análise de confusion matrix do Stage 2), **descobriu-se que o modelo Stage 2 (checkpoint `stage2_model_best.pt`) colapsou completamente**.

### Evidência Experimental

**Análise no Validation Set (38,256 samples):**

```
📊 Confusion Matrix (absoluta):
     Pred: SPLIT    RECT      AB
GT SPLIT:      0    5962       0
GT RECT :      0   17765       0
GT AB   :      0   14529       0
```

**Interpretação:** O modelo prediz **RECT (classe 1) para 100% das amostras**, independentemente da ground truth.

**Accuracy:** 46.44% (puramente devido à prevalência da classe RECT no dataset = 46.4%)

---

## 🔍 Análise do Problema

### Matriz de Confusão (Normalizada)

```json
{
  "SPLIT": {
    "correct": 0.0,
    "RECT": 1.0,    ← 100% prediz RECT
    "AB": 0.0
  },
  "RECT": {
    "correct": 1.0,  ← 100% correto (trivial)
    "SPLIT": 0.0,
    "AB": 0.0
  },
  "AB": {
    "correct": 0.0,
    "RECT": 1.0,    ← 100% prediz RECT
    "SPLIT": 0.0
  }
}
```

### Métricas por Classe

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| SPLIT  | 0.0000    | 0.0000 | 0.0000   | 5,962   |
| **RECT** | **0.4644**  | **1.0000** | **0.6342** | 17,765  |
| AB     | 0.0000    | 0.0000 | 0.0000   | 14,529  |

**Overall Accuracy:** 46.44% (= frequência da classe majoritária)

---

## ❓ Possíveis Causas

### Hipótese 1: Modelo Salvo Incorretamente ✅ PROVÁVEL
- Checkpoint pode estar corrompido ou salvo em estado intermediário ruim
- Época 8 (best model) pode ter sido antes do unfreezing começar
- Verificar: `stage2_model_final.pt` pode ter performance melhor?

### Hipótese 2: Overfitting para Classe Majoritária
- RECT = 46.4% do dataset → modelo pode ter aprendido bias forte
- CB-Focal Loss pode não ter balanceado suficientemente

### Hipótese 3: Backbone Congelado Demais
- Se modelo foi salvo com backbone frozen, pode não ter features discriminativas
- Stage 2 redesigned (Script 004) treina com backbone frozen nas primeiras épocas

### Hipótese 4: Bug na Inferência
- Problema no carregamento do estado do modelo
- Forward pass pode estar errado (mas improvável, código é simples)

---

## 🔬 Diagnóstico Necessário (URGENTE)

### Ação 1: Verificar Modelo Final ⏱️ 10 minutos

```bash
python3 pesquisa_v6/scripts/009_analyze_stage2_confusion.py \
  --stage2-model pesquisa_v6/logs/v6_experiments/stage2/stage2_model_final.pt \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --device cuda \
  --output pesquisa_v6/logs/v6_experiments/confusion_matrix_stage2_final.json
```

**Esperado:** Se final model tem F1 > 0%, então best model foi salvo no momento errado.

### Ação 2: Verificar History do Treinamento ⏱️ 5 minutos

```python
import torch
history = torch.load('pesquisa_v6/logs/v6_experiments/stage2/stage2_history.pt', weights_only=False)

print("Melhores épocas:")
print(f"Best val F1: {max(history['val_f1']):.4f} na época {history['val_f1'].index(max(history['val_f1']))}")
print(f"Best val accuracy: {max(history['val_accuracy']):.4f}")

# Ver evolução por classe
for epoch in range(len(history['val_f1'])):
    print(f"Epoch {epoch}: F1={history['val_f1'][epoch]:.4f}, Acc={history['val_accuracy'][epoch]:.4f}")
```

### Ação 3: Retreinar Stage 2 (Se Necessário) ⏱️ 2 horas

**Se ambos os modelos (best e final) estão colapsados:**

```bash
python3 pesquisa_v6/scripts/004_train_stage2_redesigned.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --epochs 30 \
  --batch-size 128 \
  --output-dir pesquisa_v6/logs/v6_experiments/stage2_retrain \
  --device cuda
```

**Modificações necessárias:**
1. Adicionar early stopping baseado em **F1 macro** (não só val loss)
2. Salvar checkpoints a cada época
3. Monitorar F1 por classe durante treinamento

---

## ⚠️ Impacto no Experimento 10

### Bloqueio Total

**O Experimento 10 (Confusion-Based Noise Injection) DEPENDE de:**
1. Matriz de confusão **realista** do Stage 2
2. Probabilidades de erro reais (não triviais: 100% → RECT)

**Com modelo colapsado:**
- Confusion matrix é inútil: todas as confusões são "X → RECT"
- Noise injection seria: "injetar 100% de RECT em Stage 3"
- **NÃO RESOLVE o problema de distribution shift**

### Risco para Experimento 09

**Se Stage 2 já estava colapsado durante Exp 09:**
- Resultados do pipeline (45.86% accuracy) podem estar errados
- Stage 3 nunca recebeu inputs diversificados (só RECT predictions)
- **TODO o Exp 09 precisaria ser refeito**

---

## 🎯 Decisão Crítica (HOJE - 13/10/2025 à noite)

### Opção A: Diagnosticar e Corrigir (2-4 horas)
1. Verificar `stage2_model_final.pt` (10 min)
2. Analisar history (5 min)
3. Se necessário: Retreinar Stage 2 (2h)
4. Revalidar Exp 09 (30 min)

**Cronograma afetado:** Exp 10 atrasa 1 dia (início 15/10 em vez de 14/10)

### Opção B: Investigar Raiz do Problema (4-8 horas - RECOMENDADO)
1. Analisar por que Stage 2 colapsou (pode ser bug sistemático)
2. Verificar se é problema de arquitetura ou treinamento
3. Implementar salvamento de checkpoints robusto
4. Retreinar com monitoramento detalhado

**Cronograma afetado:** Exp 10 atrasa 2 dias (início 16/10)

### Opção C: Usar Modelo Alternativo (Exp 11 direto)
- Pular Exp 10 (Confusion-Based Noise)
- Ir direto para Exp 11 (Train-with-Predictions com Stage 2 real-time)
- Gerar predições Stage 2 durante treinamento Stage 3

**Cronograma:** Mantém prazo (início 14/10), mas muda estratégia

---

## 📝 Ações Imediatas (HOJE - 13/10/2025)

### 1. Verificar Model Final (AGORA - 10 min) 🔴

```bash
python3 pesquisa_v6/scripts/009_analyze_stage2_confusion.py \
  --stage2-model pesquisa_v6/logs/v6_experiments/stage2/stage2_model_final.pt \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --device cuda \
  --output pesquisa_v6/logs/v6_experiments/confusion_matrix_stage2_final.json
```

**Se F1 > 30%:** Usar model_final para Exp 10  
**Se F1 ≈ 46%:** Model está OK, problema foi no salvamento do "best"  
**Se F1 ≈ 0%:** Ambos colapsaram, precisa retreinar

### 2. Analisar History (AGORA - 5 min) 🔴

```bash
python3 -c "
import torch
h = torch.load('pesquisa_v6/logs/v6_experiments/stage2/stage2_history.pt', weights_only=False)
print('Validation F1 por época:', h['val_f1'])
print('Best F1:', max(h['val_f1']), 'na época', h['val_f1'].index(max(h['val_f1'])))
"
```

### 3. Documentar Findings (20 min) 🟡

Criar: `pesquisa_v6/docs_v6/10_stage2_collapse_analysis.md`

---

## 🔗 Arquivos Relacionados

- **Script 009:** `pesquisa_v6/scripts/009_analyze_stage2_confusion.py`
- **Confusion Matrix:** `pesquisa_v6/logs/v6_experiments/confusion_matrix_stage2.json`
- **Model Best:** `pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt` (COLAPSADO)
- **Model Final:** `pesquisa_v6/logs/v6_experiments/stage2/stage2_model_final.pt` (STATUS DESCONHECIDO)
- **History:** `pesquisa_v6/logs/v6_experiments/stage2/stage2_history.pt`

---

**⚠️ PRÓXIMA AÇÃO:** Executar diagnóstico do `stage2_model_final.pt` **IMEDIATAMENTE**.

**Decisão pendente:** Qual opção (A, B ou C) seguir após diagnóstico.
