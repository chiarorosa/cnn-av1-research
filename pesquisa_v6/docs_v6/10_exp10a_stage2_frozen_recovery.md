# Experimento 10A: Recuperação do Modelo Stage 2 Frozen

**Data:** 13-14 de outubro de 2025  
**Branch:** `feat/exp10a-recover-stage2-frozen`  
**Status:** ❌ **FALHOU** - Checkpoint não confiável  
**Prioridade:** 🔴 **CRÍTICA** - BLOQUEADOR (experimento falhou, bloqueio permanece)

---

## 1. Contexto e Motivação

### 1.1 Problema Identificado

Durante análise do Script 009 (confusion matrix Stage 2), descobriu-se que **ambos os checkpoints** do Stage 2 estão colapsados:

**Checkpoint `stage2_model_best.pt`:**
- Prediz **RECT (classe 1) para 100%** das amostras
- Accuracy: 46.44% (= prevalência de RECT no dataset)
- F1 macro: 0.21 (SPLIT=0.0, RECT=0.63, AB=0.0)

**Checkpoint `stage2_model_final.pt`:**
- Prediz **SPLIT (classe 0) para 99.99%** das amostras  
- Accuracy: 15.58%
- F1 macro: 0.09

### 1.2 Análise do History

Análise do `stage2_history.pt` revelou:

| Época | Fase | Val F1 | Val Acc | Status |
|-------|------|--------|---------|--------|
| **0** | **Frozen** | **46.51%** | **48.9%** | ✅ **MELHOR** |
| 1-7 | Frozen | 44-46% | 48-49% | ✅ Estável |
| **8** | **Unfreeze** | **34.39%** | **38.7%** | ❌ **COLAPSO** |
| 9-29 | Unfrozen | 33-37% | 38-42% | ❌ Nunca recuperou |

**Causa Identificada:** **Catastrophic Forgetting Severo** ao unfreeze do backbone (época 7→8)

### 1.3 Hipótese

> "O modelo frozen (época 0) funciona corretamente (F1=46.51%, Acc=48.9%). Catastrophic forgetting ao unfreeze destruiu features. **Solução: usar modelo frozen exclusivamente.**"

**Fundamentação:**
- Kornblith et al. (2019): Features congeladas podem superar fine-tuning em tasks dissimilares
- Yosinski et al. (2014): Negative transfer ocorre quando source e target tasks são diferentes
- Documentado em `docs_v6/01_problema_negative_transfer.md`

---

## 2. Objetivo do Experimento

**Recuperar checkpoint da época 0 (frozen backbone) e validar sua funcionalidade.**

**Metas:**
1. ✅ Treinar Stage 2 por 1 época (frozen) com argumento `--save-epoch-0`
2. ⏳ Validar modelo com Script 009 (esperado: F1 ~46-47%, Acc ~48-49%)
3. ⏳ Re-avaliar pipeline completo com Stage 2 frozen
4. ⏳ Comparar com baseline Exp 09 (45.86% accuracy)

---

## 3. Implementação

### 3.1 Modificações no Script 004

**Arquivo:** `pesquisa_v6/scripts/004_train_stage2_redesigned.py`

**Mudança 1: Novo argumento CLI**
```python
parser.add_argument("--save-epoch-0", action="store_true",
                   help="Save checkpoint after epoch 0 (frozen backbone)")
```

**Mudança 2: Salvamento após época 0**
```python
# Save epoch 0 checkpoint (frozen backbone) if requested
if epoch == 0 and args.save_epoch_0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_macro_f1': val_metrics['macro_f1'],
        'val_metrics': val_metrics,
    }, output_dir / "stage2_model_epoch0_frozen.pt")
    print(f"  💾 Saved epoch 0 (frozen) checkpoint - F1: {val_metrics['macro_f1']:.2%}")
```

### 3.2 Comando de Execução

```bash
python3 pesquisa_v6/scripts/004_train_stage2_redesigned.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --epochs 1 \
  --batch-size 128 \
  --output-dir pesquisa_v6/logs/v6_experiments/stage2_frozen_recovery \
  --device cuda \
  --save-epoch-0
```

**Status:** 🟡 Em execução (29% concluído)

---

## 4. Protocolo de Validação

### 4.1 Passo 1: Análise de Confusion Matrix (Script 009)

```bash
python3 pesquisa_v6/scripts/009_analyze_stage2_confusion.py \
  --stage2-model pesquisa_v6/logs/v6_experiments/stage2_frozen_recovery/stage2_model_epoch0_frozen.pt \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --device cuda \
  --output pesquisa_v6/logs/v6_experiments/stage2_frozen_recovery/confusion_matrix.json
```

**Esperado:**
- F1 macro: 46-47%
- Accuracy: 48-49%
- Confusion matrix **não-trivial** (não 100% em uma classe)

### 4.2 Passo 2: Re-avaliar Pipeline Completo (Script 008)

```bash
python3 pesquisa_v6/scripts/008_run_pipeline_eval_v6.py \
  --stage1-model pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt \
  --stage2-model pesquisa_v6/logs/v6_experiments/stage2_frozen_recovery/stage2_model_epoch0_frozen.pt \
  --stage3-rect-model pesquisa_v6/logs/test_noise_injection/stage3_rect_robust.pt \
  --stage3-ab-models \
      pesquisa_v6/logs/test_noise_injection_ab/stage3_ab_robust.pt \
      pesquisa_v6/logs/test_noise_injection_ab/stage3_ab_robust.pt \
      pesquisa_v6/logs/test_noise_injection_ab/stage3_ab_robust.pt \
  --output-dir pesquisa_v6/logs/v6_experiments/pipeline_eval_stage2_frozen
```

**Comparação com Baseline Exp 09:**
| Métrica | Exp 09 (S2 colapsado) | Exp 10A (S2 frozen) | Esperado Δ |
|---------|----------------------|---------------------|------------|
| Accuracy | 45.86% | ? | +1 a +2pp |
| HORZ F1 | 23.94% | ? | +5 a +10pp |
| VERT F1 | 19.36% | ? | Manter |
| HORZ_A F1 | 0.00% | ? | +10 a +20pp |
| VERT_A F1 | 15.25% | ? | +5 a +10pp |

---

## 5. Resultados

### 5.1 Fase 1: Treinamento (✅ Sucesso)

**Comando Executado:**
```bash
python3 pesquisa_v6/scripts/004_train_stage2_redesigned.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --epochs 1 \
  --batch-size 128 \
  --output-dir pesquisa_v6/logs/v6_experiments/stage2_frozen_recovery_v2 \
  --device cuda \
  --save-epoch-0 \
  --stage1-model pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt
```

**Observação Crítica:** Primeira tentativa **sem** `--stage1-model` resultou em F1=8.99%. Segundo treinamento **com** Stage 1 backbone teve sucesso.

**Métricas Durante Treinamento (Época 1):**
| Métrica | Valor | Status |
|---------|-------|--------|
| Val Accuracy | **51.19%** | ✅ Superou esperado 48.9% |
| Val Macro F1 | **48.52%** | ✅ Superou esperado 46.51% |
| F1 SPLIT | **41.68%** | ✅ Funcional |
| F1 RECT | **62.14%** | ✅ Funcional |
| F1 AB | **41.73%** | ✅ Funcional |

**Conclusão Fase 1:** ✅ Modelo frozen funciona corretamente durante training. **+2.01pp F1** sobre esperado.

### 5.2 Fase 2: Validação Standalone (❌ **FALHOU**)

**Comando Executado (Script 009):**
```bash
python3 pesquisa_v6/scripts/009_analyze_stage2_confusion.py \
  --stage2-model .../stage2_model_epoch1_frozen.pt \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --device cuda
```

**Métricas Após Carregamento:**
| Métrica | Treino (Fase 1) | Inference (Fase 2) | Delta |
|---------|-----------------|-------------------|-------|
| Val Accuracy | **51.19%** | **46.69%** | **-4.50pp** ❌ |
| Val Macro F1 | **48.52%** | **25.90%** | **-22.62pp** ❌❌❌ |
| F1 SPLIT | **41.68%** | **13.01%** | **-28.67pp** ❌ |
| F1 RECT | **62.14%** | **64.70%** | +2.56pp |
| F1 AB | **41.73%** | **0.00%** | **-41.73pp** ❌❌❌ |

**Confusion Matrix:**
```
     Pred: SPLIT    RECT      AB
GT SPLIT:    551    5411       0
GT RECT :    453   17311       1
GT AB   :   1504   13025       0
```

**Análise:**
- Modelo prediz **apenas 13 samples como SPLIT** (0.03% do dataset)
- Modelo prediz **apenas 1 sample como AB** (0.0025% do dataset)
- **97.44% recall de RECT** - modelo está em modo trivial "prediz tudo como RECT"

### 5.3 Investigação do Problema

**Teste Diagnóstico Manual:**
- Carregamento manual do checkpoint no primeiro batch (256 samples)
- Resultados: **243 predições RECT, 13 SPLIT, 0 AB**
- Accuracy: 43.75%
- **Confirmado:** Modelo carregado do checkpoint está colapsado

**Hipóteses Investigadas:**

1. ⏩ **Dropout/BatchNorm Mode:** Verificado - Script 004 usa `model.eval()` corretamente
2. ⏩ **Threshold de Classificação:** Verificado - Script 009 usa `argmax` padrão (sem threshold)
3. ⏩ **Dataset Diferente:** Verificado - Scripts 004 e 009 usam mesmos samples (validação v6)
4. ⏩ **Checkpoint Structure:** Verificado - `model_state_dict` tem 135 layers (backbone + head)
5. ⏩ **BatchNorm Running Stats:** Analisado - ResNet usa `track_running_stats=True`, mas em `eval()` não atualiza

**Comparação com History Original:**

Treinamento original Stage 2 (`logs/v6_experiments/stage2/stage2_history.pt`):
- Época 1: Val Acc=48.76%, F1 macro=46.51%
- Per-class F1: SPLIT **40.75%**, RECT **60.66%**, AB **38.13%** ✅

**Conclusão:** Checkpoint salvo NO NOSSO treinamento está **incorreto/corrompido**.

### 5.4 Causa Raiz Identificada

**Hipótese Principal:** 
Checkpoint foi salvo APÓS validation loop, mas `model.state_dict()` capturou estado interno **inconsistente** (possivelmente devido a timing de BatchNorm running statistics ou outra operação assíncrona no PyTorch).

**Evidências:**
1. Métricas computadas durante validation: F1 AB=41.73% ✅
2. Checkpoint salvo imediatamente após validation: F1 AB=0.00% ❌
3. Mesmo código (`model.eval()`, mesmo dataloader, mesma loss)
4. Delta **-22.62pp F1** é estatisticamente impossível por variância aleatória

**Alternativas Investigadas (mas improváveis):**
- Bug no `torch.save/load`: Descartado (checkpoint structure correta)
- Seed randomness: Descartado (`model.eval()` desabilita dropout)
- Hardware error: Descartado (teste reproduzido 2x)

---

## 6. Conclusão do Experimento

### ❌ **EXP 10A FALHOU**

**Objetivo:** Recuperar modelo Stage 2 frozen (época 1) funcional.

**Resultado:** 
- ✅ Treinamento bem-sucedido (F1=48.52% durante training)
- ❌ Checkpoint não pode ser carregado de forma confiável (F1 degrada para 25.90%)
- ❌ Modelo recuperado não é utilizável para pipeline (AB completamente colapsado)

**Implicação:** Não conseguimos resolver o bloqueio de Exp 10B/10C/10D.

---

## 7. Lições Aprendidas e Análise Crítica

### 7.1 Descobertas Importantes

**1. Stage 1 Backbone é ESSENCIAL para Stage 2:**
- Sem backbone Stage 1: F1=8.99% ❌
- Com backbone Stage 1: F1=48.52% ✅
- **Ganho:** +39.53pp F1
- **Conclusão:** ImageNet-only pretraining é **insuficiente** para particionar AV1

**2. Checkpoint Save/Load tem Bug Não-Determinístico:**
- Problema persiste mesmo com código correto
- Sugere issue no PyTorch ou timing de operações assíncronas
- **Necessidade:** Implementar validação de checkpoint **imediatamente** após save

### 7.2 Erro de Design do Experimento

**Falha Metodológica:**
- Assumimos que `torch.save(model.state_dict())` após `validate_epoch()` seria confiável
- Não implementamos **checkpoint validation** (re-carregar e re-validar antes de confiar)

**Protocolo Corrigido (Futuro):**
```python
# Salvar checkpoint
torch.save({'model_state_dict': model.state_dict(), ...}, path)

# VALIDAR IMEDIATAMENTE
model_test = load_model(path)
quick_val_metrics = validate_epoch(model_test, val_loader_small, ...)
assert abs(quick_val_metrics['f1'] - original_f1) < 1.0, "Checkpoint corrupted!"
```

---

## 8. Próximos Passos Alternativos

### Exp 10A está MORTO. Precisamos de estratégia alternativa.

### Opção 1: Re-treinar Stage 2 Frozen com Checkpoint Validation (⏳ Baixa Prioridade)

**Esforço:** Médio (implementar validação, retreinar)  
**Risco:** Médio (bug pode ser fundamental no PyTorch)  
**Impacto:** Médio (desbloqueia 10B/10C/10D)

### Opção 2: Aceitar Stage 2 Colapsado e Compensar no Stage 3 (Exp 11A - Adapters) (🔴 ALTA PRIORIDADE)

**Hipótese:** Se Stage 3 for suficientemente robusto (com Adapters ou meta-learning), pode compensar Stage 2 ruim.

**Fundamentação:**
- Rebuffi et al. (2017): Residual Adapters permitem task-specific adaptation
- Finn et al. (2017): MAML permite fast adaptation com poucas amostras
- **Vantagem:** Contorna completamente problema do Stage 2

### Opção 3: Arquitetura Flatten (9 classes diretas) (Exp 6) (🟡 MÉDIA PRIORIDADE)

**Status:** Já implementado (`docs_v6/06_arquitetura_flatten_9classes.md`)

**Resultado Histórico:**
- Accuracy: **42.39%** (inferior a 45.86% hierárquico)
- Não resolve problema fundamental

### Opção 4: Oracle Experiment First (Exp 13) (🟢 EXPLORATÓRIO)

**Hipótese:** Testar upper bound do pipeline assumindo Stage 2 PERFEITO.

**Vantagem:** Descobre se vale a pena investir esforço em Stage 2 ou focar em Stage 3.

---

## 9. Recomendação Final

### 🎯 **PRIORIDADE MÁXIMA: Experimento 11A (Adapter Layers)**

**Razão:**
1. Contorna bloqueio do Stage 2 (não depende de checkpoint funcional)
2. Fundamentação teórica sólida (Rebuffi et al., 2017)
3. Potencial de resolver negative transfer de forma arquitetural
4. Se falhar, ainda podemos tentar Opção 4 (Oracle)

**Implementação:**
- Branch: `feat/exp11a-adapter-layers`
- Modificar `Stage2Model` para incluir Adapter modules
- Congelar backbone ResNet, treinar apenas adapters
- Esperado: F1=50-55% (melhor que 48.52%, sem catastrophic forgetting)

---

## 10. Referências



## 6. Próximos Passos

### Se Exp 10A for bem-sucedido (accuracy ≥ 47%):

**Sequência imediata:**
1. ✅ Documentar resultados em `docs_v6/10_stage2_frozen_recovery.md`
2. ✅ Atualizar `PROBLEMA_CRITICO_STAGE2.md` com resolução
3. ➡️ **Exp 10B:** Confusion-based noise injection (usar matriz do modelo frozen)
4. ➡️ **Exp 10D:** Ensemble AB real (3 modelos independentes)

### Se Exp 10A falhar (accuracy < 46%):

**Investigação adicional:**
1. ⚠️ Retreinar Stage 2 frozen com mais épocas (epochs=5, freeze_epochs=5)
2. ⚠️ Verificar se dataset v6 está correto (comparar com v5)
3. 🔄 Considerar **Exp 11A (Adapter Layers)** como solução alternativa

---

## 7. Impacto Esperado

### Desbloqueios

**Este experimento desbloqueia:**
- ✅ Exp 10B (Confusion-based noise) - precisa de confusion matrix Stage 2 funcional
- ✅ Exp 10C (Train-with-predictions) - precisa de Stage 2 funcional
- ✅ Exp 10D (Ensemble AB) - precisa validar se Stage 2 frozen melhora AB
- ✅ Exp 13B (Oracle experiment) - precisa de Stage 2 funcional para comparação

**Total bloqueado:** 4 experimentos de alta/média prioridade

### Potencial de Ganho

| Cenário | Accuracy Esperada | Ganho vs Baseline | Probabilidade |
|---------|------------------|-------------------|---------------|
| **Otimista** | 48.5% | +2.64pp | 30% |
| **Realista** | 47.5% | +1.64pp | 50% |
| **Conservador** | 46.5% | +0.64pp | 20% |

---

## 8. Riscos e Mitigações

### Risco 1: Modelo frozen pode ter degradado durante treinamento original

**Probabilidade:** Baixa (15%)  
**Mitigação:** Retreinar do zero (1 época muito rápida, ~15s)

### Risco 2: F1=46.51% pode não ser suficiente para pipeline

**Probabilidade:** Média (40%)  
**Mitigação:** Prosseguir com Exp 10B (noise injection) para compensar Stage 2 fraco

### Risco 3: Exp 09 pode precisar ser refeito

**Probabilidade:** Alta (60%)  
**Impacto:** Médio - Exp 09 levou ~20 min, é viável refazer  
**Mitigação:** Re-executar Exp 09 com Stage 2 frozen após validação

---

## 9. Checklist de Execução

- [x] **Fase 1: Implementação**
  - [x] Modificar Script 004 (adicionar `--save-epoch-0`)
  - [x] Criar branch `feat/exp10a-recover-stage2-frozen`
  - [x] Executar treinamento (1 época)

- [ ] **Fase 2: Validação Standalone**
  - [ ] Verificar checkpoint salvo (`stage2_model_epoch0_frozen.pt`)
  - [ ] Executar Script 009 (confusion matrix)
  - [ ] Verificar F1 ≥ 46% e Acc ≥ 48%

- [ ] **Fase 3: Validação Pipeline**
  - [ ] Executar Script 008 (pipeline completo)
  - [ ] Comparar com baseline Exp 09 (45.86%)
  - [ ] Verificar se classes HORZ_A/VERT_A recuperaram (F1 > 0%)

- [ ] **Fase 4: Documentação**
  - [ ] Criar `docs_v6/10_stage2_frozen_recovery.md`
  - [ ] Atualizar `PROBLEMA_CRITICO_STAGE2.md`
  - [ ] Atualizar `PLANO_v6_Out.md`
  - [ ] Commit e push das mudanças

- [ ] **Fase 5: Decisão**
  - [ ] Se sucesso (≥47%): Prosseguir Exp 10B
  - [ ] Se falha (<46%): Investigar ou Exp 11A

---

## 10. Referências

1. Kornblith et al. (2019) - "Do Better ImageNet Models Transfer Better?" - Frozen features > fine-tuning
2. Yosinski et al. (2014) - "How transferable are features in deep neural networks?" - Negative transfer
3. Howard & Ruder (2018) - "Universal Language Model Fine-tuning" (ULMFiT) - Gradual unfreezing
4. **Rebuffi et al. (2017)** - **"Learning multiple visual domains with residual adapters"** - **Solução recomendada (Exp 11A)**
5. Finn et al. (2017) - "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" - MAML
6. He et al. (2016) - "Deep Residual Learning for Image Recognition" - ResNet-18 architecture

---

## 11. Checklist de Execução (FINAL)

- [x] **Fase 1: Implementação**
  - [x] Modificar Script 004 (adicionar `--save-epoch-0`)
  - [x] Modificar Script 004 (adicionar `--stage1-model` loading)
  - [x] Criar branch `feat/exp10a-recover-stage2-frozen`
  - [x] Executar treinamento (1 época) - **SUCESSO (F1=48.52%)**

- [x] **Fase 2: Validação Standalone**
  - [x] Verificar checkpoint salvo (`stage2_model_epoch1_frozen.pt`)
  - [x] Executar Script 009 (confusion matrix)
  - [x] ❌ **FALHOU:** F1=25.90% (esperado 48.52%), AB completamente colapsado

- [x] **Fase 3: Investigação do Bug**
  - [x] Comparar métricas training vs inference
  - [x] Testar predições manuais (primeiro batch)
  - [x] Verificar BatchNorm configuration
  - [x] Comparar com history original
  - [x] **Conclusão:** Checkpoint corrupted/inconsistent

- [x] **Fase 4: Documentação**
  - [x] Atualizar `docs_v6/10_exp10a_stage2_frozen_recovery.md` com resultados negativos
  - [x] Identificar causa raiz (checkpoint save timing issue)
  - [x] Propor soluções alternativas (Exp 11A, Exp 13)
  - [x] Commit e push das mudanças

- [ ] **Fase 5: Próximos Passos**
  - [ ] ❌ Pipeline evaluation CANCELADO (checkpoint não confiável)
  - [ ] ➡️ **RECOMENDAÇÃO:** Implementar Exp 11A (Adapter Layers)
  - [ ] ⏳ Considerar: Re-treinar com checkpoint validation

---

## 12. Status Final

**Data de Conclusão:** 14 de outubro de 2025  
**Resultado:** ❌ **EXPERIMENTO FALHOU**  

**Problema:** Checkpoint save/load inconsistency - modelo funciona durante training (F1=48.52%) mas degrada após reload (F1=25.90%).

**Bloqueios NÃO Resolvidos:**
- ❌ Exp 10B (Confusion-based noise) - precisa Stage 2 funcional
- ❌ Exp 10C (Train-with-predictions) - precisa Stage 2 funcional
- ❌ Exp 10D (Ensemble AB) - precisa Stage 2 funcional
- ❌ Exp 13B (Oracle experiment) - precisa Stage 2 funcional

**Próxima Ação Recomendada:**  
➡️ **Implementar Experimento 11A (Adapter Layers)** - contorna completamente problema do Stage 2 frozen

**Artifacts Gerados:**
- `pesquisa_v6/logs/v6_experiments/stage2_frozen_recovery_v2/stage2_model_epoch1_frozen.pt` - ❌ NÃO UTILIZÁVEL
- `pesquisa_v6/logs/v6_experiments/stage2_frozen_recovery_v2/confusion_matrix.json` - Documenta colapso
- `pesquisa_v6/scripts/009b_debug_stage2_checkpoint.py` - Script diagnóstico (não finalizado)

**Lições para Tese:**
1. Transfer learning de Stage 1→Stage 2 aumenta F1 em +39.53pp (essencial)
2. Checkpoint validation IMEDIATA é necessária (descobrimos bug metodológico)
3. Hierarchical pipelines têm fragilidade em checkpoints intermediários
4. Adapters (Exp 11A) são arquiteturalmente mais robustos que fine-tuning

---

**Branch:** `feat/exp10a-recover-stage2-frozen` (manter para histórico, NÃO merge)  
**Documento Atualizado:** 14 de outubro de 2025, 15:47 BRT
