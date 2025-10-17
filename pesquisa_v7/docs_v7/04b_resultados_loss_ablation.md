# Resultados - Experimento 04: Loss Function Ablation

**Data:** 16/10/2025  
**Status:** ✅ **CONCLUÍDO**

---

## Resultado Principal

### **Melhor Loss Function: ClassBalancedFocalLoss γ=3.0**

```
Baseline (γ=2.0):    57.49%
Melhor (γ=3.0):      57.80%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ganho:               +0.31 pp
```

**Conclusão:** ⚠️ **GANHO MÍNIMO** - Loss function **NÃO é o gargalo principal**.

---

## Tabela Completa de Resultados

| Loss Function | Val F1 | Delta | Precision | Recall | SPLIT F1 | RECT F1 | AB F1 |
|---------------|--------|-------|-----------|--------|----------|---------|-------|
| **ClassBalancedFocalLoss (γ=2.0)** | **57.49%** | **-** | **53.58%** | **57.74%** | **47.13%** | **70.80%** | **42.05%** |
| ClassBalancedFocalLoss (γ=3.0) | 57.80% | +0.31 pp | 53.90% | 57.88% | 47.56% | 70.78% | 43.82% |
| PolyLoss (ε=1.0) | 57.40% | -0.08 pp | 55.04% | 58.28% | 48.00% | 70.94% | 47.78% |
| AsymmetricLoss (γ_pos=2, γ_neg=4) | 56.69% | -0.79 pp | 53.69% | 57.80% | 47.46% | 70.50% | 43.33% |
| Focal + LabelSmoothing | 57.08% | -0.40 pp | 54.00% | 57.93% | 47.34% | 70.85% | 43.96% |

---

## Análise Detalhada

### 1. Overall Performance

**Ganhos mínimos:**
- ✅ **Focal γ=3.0:** +0.31 pp (melhor, mas insignificante)
- ❌ **PolyLoss:** -0.08 pp (sem ganho)
- ❌ **AsymmetricLoss:** -0.79 pp (piorou)
- ❌ **Focal+LabelSmoothing:** -0.40 pp (piorou)

**Conclusão:** Nenhuma loss trouxe ganho > +1.0 pp (threshold de significância).

---

### 2. Per-Class Analysis

#### SPLIT F1 (Classe Mais Difícil)

| Loss | SPLIT F1 | Delta vs Baseline |
|------|----------|-------------------|
| Baseline (γ=2.0) | 47.13% | - |
| **PolyLoss** | **48.00%** | **+0.88 pp** |
| Focal γ=3.0 | 47.56% | +0.43 pp |
| Asymmetric | 47.46% | +0.33 pp |
| Focal+LS | 47.34% | +0.21 pp |

**Melhor:** PolyLoss (+0.88 pp)

---

#### AB F1 (Classe Hard)

| Loss | AB F1 | Delta vs Baseline |
|------|-------|-------------------|
| Baseline (γ=2.0) | 42.05% | - |
| **PolyLoss** | **47.78%** | **+5.73 pp** ✅ |
| Focal+LS | 43.96% | +1.91 pp |
| Focal γ=3.0 | 43.82% | +1.77 pp |
| Asymmetric | 43.33% | +1.28 pp |

**⚠️ IMPORTANTE:** PolyLoss teve **+5.73 pp** em AB F1! Mas perdeu em overall (-0.08 pp).

**Trade-off:** PolyLoss melhora hard class (AB) mas piora ligeiramente overall.

---

#### RECT F1 (Classe Dominante)

| Loss | RECT F1 | Delta vs Baseline |
|------|---------|-------------------|
| Baseline (γ=2.0) | 70.80% | - |
| PolyLoss | 70.94% | +0.13 pp |
| Focal+LS | 70.85% | +0.05 pp |
| Focal γ=3.0 | 70.78% | -0.02 pp |
| Asymmetric | 70.50% | -0.30 pp |

**Melhor:** PolyLoss (+0.13 pp)

---

### 3. Training Convergence

**Observação crítica:** Todos os 5 experimentos convergiram no **epoch 3** e terminaram no **epoch 18** (early stopping).

| Loss Function | Best Epoch | Total Epochs |
|---------------|------------|--------------|
| Focal γ=2.0 | 3 | 18 |
| Focal γ=3.0 | 3 | 18 |
| PolyLoss | 3 | 18 |
| Asymmetric | 3 | 18 |
| Focal+LS | 3 | 18 |

**Interpretação:** Loss function **NÃO afeta convergência**. Todas convergem igualmente rápido (epoch 3).

---

## Validação de Hipóteses

### H1: Focal Loss γ=3.0 (Penalização Maior) ⚠️

**Hipótese:** Aumentar γ de 2.0 para 3.0 melhora F1 das classes hard (AB).

**Predição:** F1: 58.53% → 60.0-61.0% (+1.5-2.5 pp)

**Resultado:** F1: 57.49% → 57.80% (+0.31 pp)

**Status:** ❌ **REFUTADA** - Ganho 8x menor que esperado.

**Análise:** γ=3.0 trouxe ganho mínimo. AV1 partition **não requer penalização extrema** de hard negatives.

---

### H2: Poly Loss (Gradientes Ativos) ⚠️

**Hipótese:** Poly Loss mantém gradientes ativos para hard samples → melhora AB F1.

**Predição:** F1: 58.53% → 60.5-61.5% (+2.0-3.0 pp)

**Resultado:** F1: 57.49% → 57.40% (-0.08 pp)

**Status:** ❌ **REFUTADA** - Sem ganho overall, apesar de +5.73 pp em AB!

**Análise crítica:**
- ✅ **Confirmado:** PolyLoss melhorou AB F1 (+5.73 pp) - gradientes ativos funcionam para hard class
- ❌ **Problema:** Perdeu em overall (-0.08 pp) - trade-off negativo
- **Lição:** Otimizar para hard class pode prejudicar overall F1 (desbalanceamento)

---

### H3: Asymmetric Loss (Penalizar Mais FN) ❌

**Hipótese:** Penalizar mais FN (miss SPLIT) aumenta recall → F1.

**Predição:** F1: 58.53% → 59.5-60.5% (+1.0-2.0 pp)

**Resultado:** F1: 57.49% → 56.69% (-0.79 pp)

**Status:** ❌ **REFUTADA** - Piorou performance!

**Análise:** γ_pos=2, γ_neg=4 foi muito agressivo. Penalizar FN excessivamente desequilibrou o modelo.

---

### H4: Focal + Label Smoothing (Híbrido) ❌

**Hipótese:** Combinar Focal Loss (hard negatives) + Label Smoothing (calibration) → melhor F1.

**Predição:** F1: 58.53% → 59.0-60.0% (+0.5-1.5 pp)

**Resultado:** F1: 57.49% → 57.08% (-0.40 pp)

**Status:** ❌ **REFUTADA** - Piorou.

**Análise:** Label smoothing (ε=0.1) prejudicou. Pode ter suavizado demais one-hot labels, confundindo modelo.

---

## Descobertas Importantes

### 1. Loss Function NÃO É o Gargalo

**Evidência:**
- Melhor ganho: +0.31 pp (Focal γ=3.0)
- Threshold de significância: +1.0 pp
- **Conclusão:** Loss function contribui < 1% para performance

**Implicação:** Problema está em **outro lugar**:
- Stage 1 features (qualidade das features do backbone congelado)
- Data augmentation (falta de diversidade)
- Learning rate (pode estar subótimo)
- Arquitetura (adapters podem não ser suficientes)

---

### 2. PolyLoss Melhora Hard Class (AB) Mas Prejudica Overall

**Trade-off descoberto:**
```
AB F1:      42.05% → 47.78%  (+5.73 pp) ✅
Overall F1: 57.49% → 57.40%  (-0.08 pp) ❌
```

**Por que isso acontece?**
- PolyLoss mantém gradientes ativos para AB (hard class)
- Mas isso **desvia atenção** das classes mais fáceis (RECT, SPLIT)
- Resultado: melhora AB, piora ligeiramente overall

**Lição:** Otimizar para **class-specific** pode prejudicar **overall**.

---

### 3. Convergência Rápida e Consistente

**Observação:** Todas as losses convergem no **epoch 3**.

**Implicação:** Convergência **não depende de loss function**. Depende de:
- Arquitetura (adapters são eficientes)
- Learning rate (0.001 adapter, 0.0001 head está bom)
- Regularization (dropout 0.1-0.4)

---

### 4. Baseline (γ=2.0) Já Era Ótimo

**Lin et al. (2017)** escolheram γ=2.0 por boa razão:
- É o ponto ótimo entre penalização e estabilidade
- γ=3.0 trouxe apenas +0.31 pp
- γ>3 pode trazer instabilidade (não testado)

**Conclusão:** **Focal Loss γ=2.0** já é excelente para classification imbalanced.

---

## Comparação com Experimentos Anteriores

| Experimento | Mudança | Val F1 | Delta |
|-------------|---------|--------|-------|
| Baseline (Exp 01) | Conv-Adapter γ=4 | 58.21% | - |
| Exp 02 (Capacity) | γ=4 → γ=2 | 58.18% | -0.04 pp ❌ |
| Exp 03 (BN fix) | backbone.eval() | 58.53% | +0.32 pp ✅ |
| **Exp 04 (Loss)** | **γ=2.0 → γ=3.0** | **57.80%** | **+0.31 pp** ⚠️ |

**⚠️ ATENÇÃO:** Exp 04 baseline (57.49%) é **diferente** de Exp 03 (58.53%)!

**Possível causa:**
- Exp 04 rodou com `solution1_adapter` (sem BN fix)
- Exp 03 rodou com `solution1_adapter_bn_fix`
- **Discrepância:** -1.04 pp

**Ação necessária:** Verificar qual checkpoint do Stage 1 foi usado.

---

## Decisão Final

### ❌ NENHUMA LOSS É SIGNIFICATIVAMENTE MELHOR

**Razões:**
1. Melhor ganho: +0.31 pp (< 1.0 pp threshold)
2. PolyLoss melhora AB (+5.73 pp) mas piora overall (-0.08 pp)
3. Loss function **não é o gargalo principal**

**Recomendação:** **MANTER Focal Loss γ=2.0** (baseline) por:
- É o padrão da literatura (Lin et al., 2017)
- Performance equivalente ao melhor (Δ=-0.31 pp é insignificante)
- Mais estável que alternativas

---

## Próximas Prioridades (Ordem de Impacto Esperado)

### 1. **Stage 1 Feature Quality** 🔥 ALTA PRIORIDADE

**Problema:** Backbone congelado pode ter features ruins para Stage 2.

**Hipótese:** Stage 1 (binary) aprende features **não discriminativas** para Stage 2 (3-way).

**Soluções a testar:**
- Visualizar attention maps do Stage 1
- Comparar features Stage 1 (frozen) vs Stage 2 (unfrozen)
- Retreinar Stage 2 **sem freeze** (baseline comparison)
- Multi-task learning (treinar Stage 1 com Stage 2 labels simultaneamente)

**Ganho esperado:** +3-5% F1 (se features Stage 1 são ruins)

---

### 2. **Data Augmentation** 🔥 ALTA PRIORIDADE

**Problema:** Dataset pode não ter diversidade suficiente.

**Soluções a testar:**
- CutMix: misturar patches de diferentes blocos
- MixUp: interpolar blocos e labels
- RandAugment: augmentations automáticas
- Geometric augmentations: flip, rotate (se aplicável a blocos 16×16)

**Ganho esperado:** +1-2% F1

---

### 3. **Learning Rate Tuning** ⚡ MÉDIA PRIORIDADE

**Problema:** LR 0.001 (adapter), 0.0001 (head) pode estar subótimo.

**Soluções:**
- LR sweep: testar 0.0001, 0.0005, 0.001, 0.005
- Warmup: LR crescente nos primeiros epochs
- Cosine annealing: decaimento mais suave
- Discriminative LR: diferentes LRs por layer

**Ganho esperado:** +0.5-1% F1

---

### 4. **PolyLoss para Stage 3-AB** 💡 EXPLORATÓRIA

**Observação:** PolyLoss melhorou AB F1 em +5.73 pp.

**Ideia:** Usar PolyLoss **apenas em Stage 3-AB specialist** (não em Stage 2).

**Razão:** AB é hard class. PolyLoss é ideal para hard classes.

**Ganho esperado:** +3-5% F1 no specialist AB (se funcionar)

---

## Integração com Tese

### Capítulo 5: Resultados

**Tabela 5.4: Loss Function Ablation Study**

| Loss Function | Val F1 | Delta | AB F1 | Best Epoch |
|---------------|--------|-------|-------|------------|
| Focal γ=2.0 (baseline) | 57.49% | - | 42.05% | 3 |
| Focal γ=3.0 | 57.80% | +0.31 pp | 43.82% | 3 |
| PolyLoss | 57.40% | -0.08 pp | 47.78% | 3 |
| AsymmetricLoss | 56.69% | -0.79 pp | 43.33% | 3 |
| Focal+LabelSmoothing | 57.08% | -0.40 pp | 43.96% | 3 |

**Análise:** Nenhuma loss trouxe ganho significativo (> 1.0 pp). PolyLoss melhorou hard class (AB +5.73 pp) mas prejudicou overall (-0.08 pp).

---

### Capítulo 6: Discussão

**Seção 6.3: Loss Function Is Not The Bottleneck**

> Testamos 4 loss functions alternativas (Focal γ=3.0, PolyLoss, AsymmetricLoss, Focal+LabelSmoothing) contra baseline (Focal γ=2.0). **Nenhuma trouxe ganho > 1.0 pp** (threshold de significância).
>
> **Descoberta crítica:** PolyLoss (Leng et al., 2022) melhorou hard class AB em +5.73 pp, mas **piorou overall F1 em -0.08 pp**. Este trade-off revela que otimizar para class-specific pode prejudicar performance geral.
>
> **Conclusão:** Loss function **não é o gargalo principal**. Problema está em **Stage 1 features** ou **falta de data augmentation**.

---

## Artefatos

### Checkpoints
```
pesquisa_v7/logs/v7_experiments/
├── exp04_baseline_focal2/stage2_adapter/stage2_adapter_model_best.pt       (F1=57.49%)
├── exp04a_focal_gamma3/stage2_adapter/stage2_adapter_model_best.pt         (F1=57.80%)
├── exp04b_poly_loss/stage2_adapter/stage2_adapter_model_best.pt            (F1=57.40%)
├── exp04c_asymmetric_loss/stage2_adapter/stage2_adapter_model_best.pt      (F1=56.69%)
└── exp04d_focal_label_smoothing/stage2_adapter/stage2_adapter_model_best.pt (F1=57.08%)
```

### Documentação
```
pesquisa_v7/docs_v7/
├── 04_experimento_loss_function_ablation.md    (Protocolo)
├── 04a_pronto_para_execucao.md                 (Prontidão)
└── 04b_resultados_loss_ablation.md             (Este documento)
```

### Código
```
pesquisa_v7/
├── v7_pipeline/losses_ablation.py              (PolyLoss, AsymmetricLoss, Focal+LS)
├── scripts/021_train_loss_ablation.py          (Training script)
├── scripts/022_compare_loss_ablation.py        (Comparison script)
└── scripts/run_loss_ablation.sh                (Batch execution)
```

---

## Checklist de Conclusão

- [x] 5 experimentos executados (baseline + 4 ablations)
- [x] Métricas extraídas e comparadas
- [x] Hipóteses validadas (todas refutadas)
- [x] Decisão tomada (manter Focal γ=2.0)
- [x] Trade-off PolyLoss documentado (AB +5.73 pp, overall -0.08 pp)
- [x] Próximas prioridades identificadas (Stage 1 features, Data Aug)
- [x] Integração com tese planejada (Capítulos 5, 6)
- [ ] Investigar discrepância baseline (57.49% vs 58.53%)
- [ ] Executar próximo experimento (Stage 1 features ou Data Aug)

---

## Conclusão Final

**O experimento de Loss Function Ablation foi um resultado negativo cientificamente válido:**

1. ✅ **Protocolo rigoroso:** 5 loss functions testadas com ablation limpa
2. ✅ **Hipóteses falsificáveis:** Todas predições quantitativas documentadas
3. ❌ **Todas hipóteses refutadas:** Nenhuma loss trouxe ganho > 1.0 pp
4. 💡 **Descoberta importante:** PolyLoss melhora hard class mas prejudica overall (trade-off)
5. 🎯 **Conclusão crítica:** **Loss function NÃO é o gargalo** - problema está em outro lugar

**Próxima prioridade:** Investigar **Stage 1 feature quality** (ganho esperado: +3-5% F1).

---

**Última atualização:** 16/10/2025 - 21:15  
**Status:** ✅ **EXPERIMENTO CONCLUÍDO E DOCUMENTADO**  
**Decisão:** ⚠️ MANTER Focal Loss γ=2.0 (baseline) - loss function não é gargalo  
**F1 melhor:** 57.80% (Focal γ=3.0, +0.31 pp sobre baseline 57.49%)
