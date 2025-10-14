

# Pesquisa V7 - Soluções Finais para Tese de Doutorado

**Branch:** `pesquisa_v7`  
**Status:** Em desenvolvimento  
**Objetivo:** Implementar e comparar 3 soluções arquiteturais fundamentadas em literatura para alcançar alta acurácia na predição de partições AV1.

---

## 📚 Fundamentação Teórica

### Artigos Base

1. **Chen et al. (CVPR 2024)** - "Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets"
   - **Conceito:** Parameter Efficient Tuning (PET) - congela backbone, aprende apenas modulação
   - **Resultado:** Iguala ou supera full fine-tuning com apenas 3.5% dos parâmetros
   - **Aplicação:** Resolver negative transfer (Stage 1 → Stage 2)

2. **Ahad et al. (2024)** - "Ensemble Model for Breast Cancer Detection"
   - **Conceito:** Ensemble com voting-based (DenseNet121 + InceptionV3 + ResNet18)
   - **Resultado:** 99.94% acurácia vs 99% do melhor modelo individual
   - **Aplicação:** Compensar erros individuais em classificação hierárquica

---

## 🎯 As Três Soluções

### **Solução 1: Conv-Adapter** (Parameter Efficient Transfer)

**Literatura:** Chen et al., CVPR 2024

**Arquitetura:**
```
Stage 1: Treina backbone + binary head (NONE vs PARTITION)
         ↓ (congela backbone)
Stage 2: Conv-Adapter (3.5% params) + 3-way head (SPLIT, RECT, AB)
         ↓ (congela backbone)
Stage 3: Conv-Adapter (3.5% params) + specialist heads
```

**Mecanismo:** `h ← h + α·Δh` (feature modulation, não substituição)

**Hipótese:** Congelar backbone após Stage 1 elimina negative transfer  
**Expectativa:** Stage 2 F1: 46% → 60-65%

**Implementação:** `v7_pipeline/conv_adapter.py`

---

### **Solução 2: Multi-Stage Ensemble**

**Literatura:** Ahad et al., 2024

**Arquitetura:**
```
3 backbones diferentes (ResNet18, MobileNetV2, EfficientNet)
         ↓
Cada backbone tem pipeline hierárquica completa (Stage 1→2→3)
         ↓
Soft voting em cada estágio
```

**Mecanismo:** Weighted average de probabilidades

**Hipótese:** Diversidade arquitetural compensa erros individuais  
**Expectativa:** Pipeline F1: +5-8% vs melhor modelo individual

**Implementação:** `v7_pipeline/ensemble.py`

---

### **Solução 3: Hybrid (Adapter + Ensemble)** ⭐ **RECOMENDADO**

**Literatura:** Combinação inovadora de Chen et al. + Ahad et al.

**Arquitetura:**
```
Backbone compartilhado (frozen após Stage 1)
         ↓
3 Conv-Adapters com configs diferentes (reduction=4,8,4)
         ↓
Soft voting over adapter outputs
         ↓
Aplicado em TODOS os stages (Stage 2, 3-RECT, 3-AB)
```

**Vantagens:**
- ✅ Parameter efficient: ~10% trainable (vs 300% para 3 backbones)
- ✅ Negative transfer prevention (backbone frozen)
- ✅ Ensemble boosting (+1-5% F1)
- ✅ Few-shot robustness (adapters excelentes em classes raras)

**Expectativa:**
- Stage 2 F1: 46% → **65-73%**
- Stage 3-AB F1: 24.5% → **45-50%**
- Pipeline completo: **~70-75%**

**Implementação:** `v7_pipeline/hybrid_model.py`

---

## 📁 Estrutura do Projeto

```
pesquisa_v7/
├── README.md                          # Este arquivo
├── ARCHITECTURE_V7.md                 # Diagramas arquiteturais detalhados
│
├── scripts/
│   ├── 001_prepare_dataset.py         # Preparação base (v6 dataset)
│   ├── 002_prepare_stage3_datasets.py # Especialistas RECT/AB
│   ├── 010_baseline_v6_reproduction.py    # Baseline comparativo
│   ├── 020_train_adapter_solution.py      # Solução 1: Conv-Adapter
│   ├── 030_train_ensemble_solution.py     # Solução 2: Ensemble
│   ├── 040_train_hybrid_solution.py       # Solução 3: Híbrido
│   └── 050_evaluate_all_solutions.py      # Comparação final
│
├── v7_pipeline/
│   ├── __init__.py
│   ├── data_hub.py              # Dataset classes (copiado v6)
│   ├── losses.py                # Loss functions (copiado v6)
│   ├── backbone.py              # ImprovedBackbone refatorado
│   ├── conv_adapter.py          # NOVO - Conv-Adapter (Chen et al.)
│   ├── ensemble.py              # NOVO - Ensemble voting (Ahad et al.)
│   ├── hybrid_model.py          # NOVO - Adapter + Ensemble
│   └── evaluation.py            # NOVO - Métricas unificadas
│
├── logs/v7_experiments/
│   ├── baseline/                # Resultados v6 reproduction
│   ├── solution1_adapter/       # Conv-Adapter results
│   ├── solution2_ensemble/      # Ensemble results
│   └── solution3_hybrid/        # Hybrid results (MELHOR ESPERADO)
│
└── docs_v7/
    ├── 00_README.md             # Índice documentação
    ├── 01_solution1_conv_adapter.md
    ├── 02_solution2_ensemble.md
    ├── 03_solution3_hybrid.md
    └── 04_comparative_analysis.md
```

---

## 🚀 Workflow de Experimentos

### **Fase 1: Preparação de Dados**

```bash
# 1. Preparar dataset base (usa v6_dataset se já existe)
python3 pesquisa_v7/scripts/001_prepare_dataset.py \
  --base-path /home/chiarorosa/experimentos/uvg/ \
  --output-dir pesquisa_v7/v7_dataset

# 2. Preparar datasets Stage 3
python3 pesquisa_v7/scripts/002_prepare_stage3_datasets.py \
  --dataset-dir pesquisa_v7/v7_dataset/block_16
```

### **Fase 2: Baseline (Reprodução v6)**

```bash
# Treinar baseline para comparação
python3 pesquisa_v7/scripts/010_baseline_v6_reproduction.py \
  --dataset-dir pesquisa_v7/v7_dataset/block_16 \
  --output-dir pesquisa_v7/logs/v7_experiments/baseline
```

### **Fase 3: Solução 1 - Conv-Adapter**

```bash
# Treinar com Conv-Adapter
python3 pesquisa_v7/scripts/020_train_adapter_solution.py \
  --dataset-dir pesquisa_v7/v7_dataset/block_16 \
  --stage1-checkpoint <path_to_stage1> \
  --adapter-reduction 4 \
  --output-dir pesquisa_v7/logs/v7_experiments/solution1_adapter
```

### **Fase 4: Solução 2 - Ensemble**

```bash
# Treinar ensemble (3 modelos)
python3 pesquisa_v7/scripts/030_train_ensemble_solution.py \
  --dataset-dir pesquisa_v7/v7_dataset/block_16 \
  --num-models 3 \
  --output-dir pesquisa_v7/logs/v7_experiments/solution2_ensemble
```

### **Fase 5: Solução 3 - Híbrido** ⭐

```bash
# Treinar solução híbrida (RECOMENDADO)
python3 pesquisa_v7/scripts/040_train_hybrid_solution.py \
  --dataset-dir pesquisa_v7/v7_dataset/block_16 \
  --stage1-checkpoint <path_to_stage1> \
  --num-adapters 3 \
  --output-dir pesquisa_v7/logs/v7_experiments/solution3_hybrid
```

### **Fase 6: Avaliação Comparativa**

```bash
# Comparar todas as soluções
python3 pesquisa_v7/scripts/050_evaluate_all_solutions.py \
  --baseline-dir pesquisa_v7/logs/v7_experiments/baseline \
  --adapter-dir pesquisa_v7/logs/v7_experiments/solution1_adapter \
  --ensemble-dir pesquisa_v7/logs/v7_experiments/solution2_ensemble \
  --hybrid-dir pesquisa_v7/logs/v7_experiments/solution3_hybrid \
  --output-file pesquisa_v7/logs/v7_experiments/comparative_results.json
```

---

## 📊 Métricas de Avaliação

### **Métricas Principais:**
1. **F1-score macro** (por stage e pipeline completo)
2. **F1 per-class** (especialmente classes raras: HORZ_A/B, VERT_A/B, HORZ_4, VERT_4)
3. **Accuracy**
4. **Precision/Recall**

### **Métricas Secundárias:**
5. **Parameter efficiency** (% parâmetros treináveis)
6. **Training time** (epochs to convergence)
7. **Inference speed** (samples/second)
8. **Confusion matrix** (análise de erros)

### **Comparação Esperada:**

| Solução | Stage 2 F1 | Stage 3-AB F1 | Pipeline F1 | Trainable Params |
|---------|------------|---------------|-------------|------------------|
| Baseline v6 | 46% | 24.5% | ~65% | 100% |
| Conv-Adapter | **60-65%** | **40-45%** | **~68-72%** | **3.5%** |
| Ensemble | 50-55% | 30-35% | ~70% | 300% |
| **Hybrid** | **65-73%** | **45-50%** | **70-75%** | **~10%** |

---

## 🔬 Protocolo Experimental (PhD-Level)

### **Hipóteses Científicas:**

**H1 (Conv-Adapter):** Congelar backbone após Stage 1 e usar adapters elimina negative transfer, aumentando F1 do Stage 2 de 46% para >60%.

**H2 (Ensemble):** Diversidade arquitetural compensa erros individuais, aumentando F1 pipeline em 5-8% absoluto.

**H3 (Híbrido):** Combinação de adapters (efficiency) + ensemble (robustness) supera ambas soluções isoladas.

### **Validação:**

1. **Ablation Studies:**
   - Adapter vs Full fine-tuning vs Frozen backbone
   - Reduction ratio (γ=2, 4, 8, 16)
   - Número de adapters no ensemble (2, 3, 5)
   - Soft voting vs Hard voting

2. **Controles:**
   - Mesmo dataset split (train/val)
   - Mesmos hiperparâmetros base (lr, batch_size)
   - Mesmo seed (42) para reprodutibilidade
   - GPU fixa (mesmo hardware)

3. **Análise Estatística:**
   - 3 runs por experimento (média ± std)
   - Teste de significância (>5% melhoria = significativo)
   - Intervalos de confiança 95%

### **Artifacts:**

Para cada experimento, salvar:
- ✅ Checkpoints (best e final)
- ✅ Training history (losses, metrics por epoch)
- ✅ Confusion matrix
- ✅ Per-class metrics
- ✅ Attention maps (adapters)
- ✅ Ensemble weights (learned)

---

## 📖 Documentação para Tese

Cada solução terá documentação completa em `docs_v7/`:

### **Estrutura de cada documento:**

1. **Motivação** (2-3 parágrafos)
2. **Literatura Foundation** (5-10 papers citados)
3. **Hipótese** (testável, quantitativa)
4. **Protocolo Experimental** (reprodutível)
5. **Arquitetura Detalhada** (diagramas, equações)
6. **Resultados** (tabelas, gráficos, confusion matrices)
7. **Análise Crítica** (por que funcionou/falhou?)
8. **Limitations** (o que não foi testado?)
9. **Contribuições** (inovação PhD-level)
10. **References** (bibliografia completa)

---

## 🎓 Contribuições Acadêmicas

### **Inovações deste trabalho:**

1. **Primeira aplicação de Conv-Adapter** (Chen et al., CVPR 2024) em **predição de partições de codecs de vídeo**
   - Domínio: Computer Vision → Video Compression
   - Desafio: Negative transfer em classificação hierárquica

2. **Ensemble hierárquico multi-stage** para classificação desbalanceada
   - Não apenas ensemble no final (comum)
   - Voting em CADA estágio da hierarquia

3. **Híbrido Adapter+Ensemble** para efficiency + robustness
   - Combinação inédita na literatura
   - Trade-off otimizado: 10% params, 70-75% F1

4. **Benchmark completo** para AV1 partition prediction
   - Dataset público (UVG)
   - 3 soluções comparadas rigorosamente
   - Código reprodutível

### **Possíveis publicações:**

1. **Workshop paper** (CVPR/ICCV): "Conv-Adapter for Video Codec Partition Prediction"
2. **Journal paper** (IEEE TIP/TCSVT): "Hierarchical Ensemble Methods for AV1 Encoding Acceleration"
3. **Thesis chapter**: "Parameter-Efficient Transfer Learning for Fine-Grained Video Classification"

---

## ⚙️ Requisitos

### **Ambiente:**
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU obrigatória)
- 16GB+ RAM
- ~50GB storage (datasets + checkpoints)

### **Dependências:**
Mesmas do v6 (ver `/requirements.txt`)

---

## 📝 TODO

### **Prioridade Alta:**
- [ ] Criar scripts 010-050 (treinamento e avaliação)
- [ ] Treinar Stage 1 baseline (necessário para todas soluções)
- [ ] Implementar Solution 1 (Conv-Adapter)
- [ ] Validar adapter com ablation (reduction=4,8,16)

### **Prioridade Média:**
- [ ] Implementar Solution 2 (Ensemble)
- [ ] Testar diversidade: ResNet18 vs MobileNetV2
- [ ] Implementar Solution 3 (Hybrid)
- [ ] Grid search: num_adapters (2,3,5)

### **Prioridade Baixa:**
- [ ] Criar ARCHITECTURE_V7.md com diagramas
- [ ] Documentar experiments em docs_v7/
- [ ] Gerar visualizações (attention maps, confusion matrices)
- [ ] Preparar material para tese

---

## 🔗 Referências Rápidas

- **v6 (versão anterior):** `/pesquisa_v6/`
- **Dataset raw:** `/home/chiarorosa/experimentos/uvg/`
- **Artigos base:** `/pesquisa_v6/artigos/`
- **Copilot instructions:** `/.github/copilot-instructions.md`

---

## 👤 Autor

**Chiaro Rosa**  
PhD Candidate  
Research: Deep Learning for AV1 Video Codec Optimization

**Contact:** chiarorosa@...  
**Repository:** https://github.com/chiarorosa/cnn-av1-research

---

## 📄 Licença

Código acadêmico para pesquisa de doutorado.  
Para uso comercial, entrar em contato.

---

**Última atualização:** 14 de Outubro de 2025  
**Status:** 🚧 Em desenvolvimento ativo
