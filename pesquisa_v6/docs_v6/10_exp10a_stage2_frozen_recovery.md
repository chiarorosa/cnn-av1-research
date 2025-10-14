# Experimento 10A: Recuperação do Modelo Stage 2 Frozen

**Data:** 13 de outubro de 2025  
**Branch:** `feat/exp10a-recover-stage2-frozen`  
**Status:** 🟡 EM EXECUÇÃO  
**Prioridade:** 🔴 **CRÍTICA** - BLOQUEADOR

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

## 5. Resultados Esperados

### 5.1 Validação Stage 2 Standalone

**Hipótese H1:** Modelo frozen tem F1 ~46-47%

**Se H1 verdadeira:**
- ✅ Confirma que época 0 funciona
- ✅ Stage 2 não está completamente quebrado
- ✅ Desbloqueia Exp 10B (Confusion-based noise injection)

**Se H1 falsa (F1 < 40%):**
- ❌ Problema mais profundo que catastrophic forgetting
- ⚠️ Investigar: dataset, loss function, ou arquitetura
- 🔄 Considerar Exp 11A (Adapter Layers) como alternativa

### 5.2 Pipeline Completo

**Hipótese H2:** Pipeline accuracy 45.86% → 47-48% (+1.5pp)

**Razão:** Stage 2 frozen (F1=46.51%) distribui melhor samples para Stage 3 que Stage 2 colapsado (prediz 100% RECT)

**Se H2 verdadeira:**
- ✅ Confirma que Stage 2 colapsado era o problema
- ✅ Exp 09 (Noise Injection) estava testando com Stage 2 quebrado
- ✅ Validar se Exp 09 precisa ser refeito

**Se H2 falsa (accuracy < 46%):**
- ⚠️ Stage 2 frozen não é suficiente
- 🔄 Prosseguir com Exp 10B (Confusion-based noise) é ainda mais crítico

---

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
3. Howard & Ruder (2018) - "Universal Language Model Fine-tuning" (ULMFiT) - Gradual unfreezing falhou
4. Rebuffi et al. (2017) - "Learning multiple visual domains with residual adapters" - Alternativa futura (Exp 11A)

---

**Status Atual:** 🟡 Treinamento em progresso (29%)  
**Próxima Ação:** Aguardar conclusão do treinamento e executar Fase 2 (validação)  
**Tempo Estimado:** 10-15 minutos para treino + 5 minutos validação = **20 minutos total**
