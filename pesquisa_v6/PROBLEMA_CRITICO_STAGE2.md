# PROBLEMA CRÍTICO IDENTIFICADO: Stage 2 Model Colapsado

**Data:** 13/10/2025  
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

### Ação 1: Verificar Modelo Final

```bash
python3 pesquisa_v6/scripts/009_analyze_stage2_confusion.py \
  --stage2-model pesquisa_v6/logs/v6_experiments/stage2/stage2_model_final.pt \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --device cuda \
  --output pesquisa_v6/logs/v6_experiments/confusion_matrix_stage2_final.json
```

**Esperado:** Se final model tem F1 > 0%, então best model foi salvo no momento errado.

### Ação 2: Verificar History do Treinamento

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

### Ação 3: Retreinar Stage 2 (Se Necessário)

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

## 🎯 DECISÃO RECOMENDADA

### Primeira Prioridade: Opção A (Usar Modelo Frozen) ⭐

**Razões:**
1. ✅ **Mais rápido** (menos complexidade)
2. ✅ **Menor risco** (modelo frozen já provou F1=0.4651)
3. ✅ **Validação científica** (confirma hipótese de que frozen > unfrozen)
4. ✅ **Mantém foco** (Exp 10 pode prosseguir)

**Se Opção A falhar** (modelo frozen não disponível e retreino frozen também colapsa):
→ Tentar **Opção C** (Train-with-Predictions)
→ Pular Opção B (retreinar completo é alto custo, baixo ganho esperado)

---

## 📝 Próxima Ação Imediata

### 1. Verificar Existência de Checkpoint Frozen 🔴 URGENTE

```bash
cd /home/chiarorosa/CNN_AV1
ls -lh pesquisa_v6/logs/v6_experiments/stage2/ | grep -E "ep[0-9]|frozen"
```

**Se encontrar `stage2_model_ep0.pt` ou similar:**
✅ Executar Script 009 nele e validar F1 ~0.46

**Se NÃO encontrar:**
⚠️ Retreinar 1 época frozen

### 2. Atualizar Documentação 🟡

- ✅ PROBLEMA_CRITICO_STAGE2.md criado
- ✅ PROXIMOS_PASSOS.md atualizado
- ⏳ Criar `docs_v6/10_stage2_collapse_analysis.md` (PhD-level)

### 3. Push das Mudanças 🟡

```bash
git add pesquisa_v6/PROXIMOS_PASSOS.md .github/copilot-instructions.md
git commit -m "docs: Remover estimativas de tempo/datas dos documentos"
git push origin main
```

---

**⚠️ PRÓXIMA AÇÃO:** Executar diagnóstico do `stage2_model_final.pt` ou localizar checkpoint frozen.

**Decisão pendente:** Qual opção (A, B ou C) seguir após diagnóstico.

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
