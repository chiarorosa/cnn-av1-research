# Experimento 04: Loss Function Ablation - Prontidão Para Execução

**Data:** 16/10/2025  
**Status:** ✅ **PRONTO PARA EXECUTAR**

---

## Resumo Executivo

Implementação completa do **Experimento 04: Loss Function Ablation** para identificar a melhor loss function para Stage 2 do pipeline v7.

**Motivação:** F1 estagnou em ~58.5% após 3 experimentos (baseline, capacity, BatchNorm fix). Hipótese: **loss function é o gargalo**, não a arquitetura.

**Expectativa:** +2-3 pp de ganho (F1: 58.5% → 60.5-61.5%)

---

## Artefatos Criados

### 1. Documentação Científica

**`04_experimento_loss_function_ablation.md`** (11 seções, PhD-level)
- ✅ Motivação com literatura (Lin, Leng, Ridnik, Müller)
- ✅ 4 hipóteses testáveis com predições quantitativas
- ✅ Protocolo experimental detalhado (100% reprodutível)
- ✅ Critérios de sucesso/falha
- ✅ Análise planejada (métricas, tabelas, testes estatísticos)
- ✅ Riscos e mitigação
- ✅ Integração com tese (Capítulos 4, 5, 6)

### 2. Código Implementado

#### `v7_pipeline/losses_ablation.py` (300+ linhas)

Implementa 3 novas loss functions:

1. **PolyLoss** (Leng et al., NeurIPS 2022)
   - Cross-entropy + termo polinomial
   - Mantém gradientes ativos para hard samples
   - ε=1.0 (padrão)

2. **AsymmetricLoss** (Ridnik et al., ICCV 2021)
   - Penalidades assimétricas (FP vs FN)
   - Adaptado para multi-class (one-vs-rest)
   - γ_pos=2.0, γ_neg=4.0

3. **FocalLossWithLabelSmoothing** (híbrido)
   - Focal Loss (hard negatives) + Label Smoothing (calibration)
   - γ=2.0, ε=0.1

**Status:** ✅ Unit tests passando

```
✓ PolyLoss test passed
✓ AsymmetricLoss test passed
✓ FocalLossWithLabelSmoothing test passed
```

#### `scripts/021_train_loss_ablation.py` (600+ linhas)

Script de treinamento para ablation study:
- Baseado em `020_train_adapter_solution.py`
- **TUDO fixo** exceto loss function (ablation limpa)
- Suporta 5 loss types: `baseline`, `focal_gamma3`, `poly`, `asymmetric`, `focal_smoothing`
- BatchNorm fix aplicado: `adapter_backbone.backbone.eval()`

**Uso:**
```bash
python3 pesquisa_v7/scripts/021_train_loss_ablation.py \
  --dataset-dir pesquisa_v7/v7_dataset/block_16 \
  --stage1-checkpoint pesquisa_v7/logs/v7_experiments/solution1_adapter/stage1/stage1_model_best.pt \
  --output-dir pesquisa_v7/logs/v7_experiments/exp04b_poly_loss \
  --loss-type poly \
  --device cuda \
  --batch-size 128 \
  --epochs 50 \
  --seed 42
```

#### `scripts/run_loss_ablation.sh` (executável)

Script bash para rodar todos os 5 experimentos sequencialmente:
- Valida pré-requisitos (Stage 1 checkpoint, dataset)
- Roda 5 experimentos com mesmos hyperparâmetros
- Estima ~2h por experimento (~10h total sequencial)

**Uso:**
```bash
bash pesquisa_v7/scripts/run_loss_ablation.sh
```

#### `scripts/022_compare_loss_ablation.py` (análise)

Script Python para comparar resultados de todos os experimentos:
- Carrega métricas de todos os 5 experimentos
- Gera 3 tabelas: overall performance, per-class F1, training details
- Identifica melhor loss function
- Testes de significância (Δ > +1.0 pp)
- Gera tabela markdown para documentação

**Uso:**
```bash
python3 pesquisa_v7/scripts/022_compare_loss_ablation.py
```

---

## Experimentos Planejados

### Baseline (Referência)

**Loss:** ClassBalancedFocalLoss γ=2.0  
**F1 esperado:** 58.53% (conhecido do Exp 03)  
**Output:** `exp04_baseline_focal2/`

---

### Exp 4A: Focal Loss γ=3.0

**Hipótese:** Maior penalização de hard negatives melhora F1  
**Mudança:** γ: 2.0 → 3.0  
**Predição:** F1: 58.53% → **60.0-61.0%** (+1.5-2.5 pp)  
**Literatura:** Lin et al. (ICCV 2017)  
**Output:** `exp04a_focal_gamma3/`

---

### Exp 4B: Poly Loss

**Hipótese:** Gradientes ativos para hard samples melhoram AB F1  
**Mudança:** Cross-entropy → Poly Loss (ε=1.0)  
**Predição:** F1: 58.53% → **60.5-61.5%** (+2.0-3.0 pp)  
**Literatura:** Leng et al. (NeurIPS 2022)  
**Output:** `exp04b_poly_loss/`

---

### Exp 4C: Asymmetric Loss

**Hipótese:** Penalizar mais FN (miss SPLIT) aumenta recall → F1  
**Mudança:** Focal → Asymmetric (γ_pos=2, γ_neg=4)  
**Predição:** F1: 58.53% → **59.5-60.5%** (+1.0-2.0 pp)  
**Literatura:** Ridnik et al. (ICCV 2021)  
**Output:** `exp04c_asymmetric_loss/`

---

### Exp 4D: Focal + Label Smoothing

**Hipótese:** Combinar hard negatives + calibration → melhor F1 + ECE  
**Mudança:** Focal → Focal + Label Smoothing (ε=0.1)  
**Predição:** F1: 58.53% → **59.0-60.0%** (+0.5-1.5 pp)  
**Literatura:** Lin et al. + Müller et al. (NeurIPS 2019)  
**Output:** `exp04d_focal_label_smoothing/`

---

## Pré-Requisitos

### ✅ Verificados

- [x] Dataset: `pesquisa_v7/v7_dataset/block_16/` ✓ Existe
- [x] Stage 1 checkpoint: `solution1_adapter/stage1/stage1_model_best.pt` ✓ Existe
- [x] GPU disponível: CUDA ✓
- [x] Python environment: `.venv` ativado ✓
- [x] Unit tests: Todas as losses testadas ✓

### Arquivos Criados

```
pesquisa_v7/
├── v7_pipeline/
│   └── losses_ablation.py                    ✓ Implementado + testado
├── scripts/
│   ├── 021_train_loss_ablation.py            ✓ Implementado
│   ├── 022_compare_loss_ablation.py          ✓ Implementado
│   └── run_loss_ablation.sh                  ✓ Executável
└── docs_v7/
    ├── 04_experimento_loss_function_ablation.md  ✓ Protocolo completo
    └── 04a_pronto_para_execucao.md           ✓ Este documento
```

---

## Como Executar

### Opção 1: Rodar Todos Sequencialmente (Recomendado)

```bash
cd /home/chiarorosa/CNN_AV1
source .venv/bin/activate
bash pesquisa_v7/scripts/run_loss_ablation.sh
```

**Tempo estimado:** ~10h (5 experimentos × 2h cada)

---

### Opção 2: Rodar Individualmente

```bash
cd /home/chiarorosa/CNN_AV1
source .venv/bin/activate

# Exp 4A: Focal γ=3.0
python3 pesquisa_v7/scripts/021_train_loss_ablation.py \
  --dataset-dir pesquisa_v7/v7_dataset/block_16 \
  --stage1-checkpoint pesquisa_v7/logs/v7_experiments/solution1_adapter/stage1/stage1_model_best.pt \
  --output-dir pesquisa_v7/logs/v7_experiments/exp04a_focal_gamma3 \
  --loss-type focal_gamma3 \
  --device cuda \
  --batch-size 128 \
  --epochs 50 \
  --seed 42

# Exp 4B: Poly Loss
python3 pesquisa_v7/scripts/021_train_loss_ablation.py \
  --loss-type poly \
  --output-dir pesquisa_v7/logs/v7_experiments/exp04b_poly_loss \
  # ... (mesmos args)

# Exp 4C: Asymmetric Loss
python3 pesquisa_v7/scripts/021_train_loss_ablation.py \
  --loss-type asymmetric \
  --output-dir pesquisa_v7/logs/v7_experiments/exp04c_asymmetric_loss \
  # ... (mesmos args)

# Exp 4D: Focal + Label Smoothing
python3 pesquisa_v7/scripts/021_train_loss_ablation.py \
  --loss-type focal_smoothing \
  --output-dir pesquisa_v7/logs/v7_experiments/exp04d_focal_label_smoothing \
  # ... (mesmos args)
```

---

### Opção 3: Rodar em Paralelo (4 GPUs)

Se você tiver múltiplas GPUs, pode rodar 4 experimentos simultaneamente:

```bash
# GPU 0: Focal γ=3.0
CUDA_VISIBLE_DEVICES=0 python3 ... --loss-type focal_gamma3 &

# GPU 1: Poly Loss
CUDA_VISIBLE_DEVICES=1 python3 ... --loss-type poly &

# GPU 2: Asymmetric Loss
CUDA_VISIBLE_DEVICES=2 python3 ... --loss-type asymmetric &

# GPU 3: Focal + Label Smoothing
CUDA_VISIBLE_DEVICES=3 python3 ... --loss-type focal_smoothing &

wait  # Aguardar conclusão
```

**Tempo estimado:** ~2h (paralelo)

---

## Análise de Resultados

Após completar os treinamentos:

```bash
# Comparar todos os experimentos
python3 pesquisa_v7/scripts/022_compare_loss_ablation.py
```

**Output esperado:**
- Tabela 1: Overall performance (F1, precision, recall)
- Tabela 2: Per-class F1 (SPLIT, RECT, AB)
- Tabela 3: Training details (epochs, params)
- Análise: melhor loss, ganhos significativos, per-class improvements
- Markdown table (pronto para copiar em `04b_resultados_loss_ablation.md`)

---

## Critérios de Sucesso

### ✅ Sucesso Completo

- **Val F1 > 60.5%** (+2.0 pp sobre baseline 58.53%)
- AB F1 > 20% (ganho significativo em hard class)
- Training estável (sem divergência)
- Reprodutível (seed 42)

### ⚠️ Sucesso Parcial

- **Val F1 > 59.5%** (+1.0 pp)
- AB F1 aumentou (mesmo se < 20%)
- Trade-offs aceitáveis (e.g., -1% precision, +3% recall)

### ❌ Falha

- Val F1 < 59.0% (< +0.5 pp)
- AB F1 não melhorou ou piorou
- Training instável (NaN losses, divergência)

**Se falhar:** Documentar resultado negativo (PhD-level) e prosseguir para **Data Augmentation** (próxima prioridade).

---

## Próximos Passos (Após Experimento 04)

### Se Bem-Sucedido (F1 > 60%)

1. **Documentar resultados:** Criar `04b_resultados_loss_ablation.md`
2. **Atualizar tese:** Integrar com Capítulos 4 (Metodologia), 5 (Resultados), 6 (Discussão)
3. **Aplicar melhor loss em Stage 3:** Retreinar RECT e AB specialists
4. **Combinar com Data Augmentation:** Testar melhor loss + CutMix/MixUp

### Se Parcial (59% < F1 < 60.5%)

1. Documentar ganho moderado
2. Testar **Data Augmentation** (CutMix, MixUp) com melhor loss
3. Considerar **Learning Rate tuning**

### Se Falhar (F1 < 59%)

1. Documentar resultado negativo (contribuição científica: "o que NÃO funciona")
2. **Prosseguir para Data Augmentation** (ALTA PRIORIDADE)
3. Considerar que problema está em **Stage 1 features** (análise qualitativa)

---

## Checklist de Execução

### Antes de Começar

- [ ] Confirmar Stage 1 checkpoint existe e tem F1 > 60%
- [ ] Confirmar dataset `v7_dataset/block_16/` existe
- [ ] GPU disponível (8GB VRAM mínimo)
- [ ] `.venv` ativado

### Durante Treinamento

- [ ] Monitorar logs (loss/F1 a cada epoch)
- [ ] Verificar stability (sem NaN, divergência)
- [ ] Checkpoints salvando corretamente

### Após Treinamento

- [ ] Rodar `022_compare_loss_ablation.py`
- [ ] Analisar tabelas comparativas
- [ ] Identificar melhor loss function
- [ ] Documentar em `04b_resultados_loss_ablation.md`
- [ ] Decidir próximo passo (Stage 3, Data Aug, ou LR tuning)

---

## Contato e Suporte

**Pesquisador:** Chiaro Rosa  
**Projeto:** PhD Research - CNN-AV1  
**Experimento:** 04 - Loss Function Ablation  
**Data:** 16/10/2025  

**Status:** ✅ **PRONTO PARA EXECUTAR**

---

**Última atualização:** 16/10/2025 - 01:00  
**Próxima ação:** Executar `bash pesquisa_v7/scripts/run_loss_ablation.sh` ou rodar experimentos individuais
