# Arquitetura V7 - Visão Geral das Soluções

## 📐 Diagrama Conceitual

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PESQUISA V7                                   │
│            Soluções Finais para Tese de Doutorado                   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  Problema Identificado  │
                    │  (v6 documentation)     │
                    └─────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌──────────────────┐     ┌──────────────────┐
│ Negative      │      │ Class Imbalance  │     │ Stage 3-AB       │
│ Transfer      │      │ (extreme)        │     │ Collapse         │
│               │      │                  │     │                  │
│ Stage2 F1:    │      │ HORZ_A: 20%     │     │ F1 = 24.5%       │
│ 46% → 32%     │      │ VERT_B: 35%     │     │ (4 classes)      │
└───────────────┘      └──────────────────┘     └──────────────────┘
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Literatura Review      │
                    │  (2 artigos)            │
                    └─────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌──────────────────┐     ┌──────────────────┐
│ SOLUÇÃO 1     │      │ SOLUÇÃO 2        │     │ SOLUÇÃO 3        │
│ Conv-Adapter  │      │ Multi-Ensemble   │     │ HÍBRIDO          │
│               │      │                  │     │ (Adapter+Ensemble)│
│ Chen+ CVPR'24 │      │ Ahad+ 2024       │     │ NOVEL            │
└───────────────┘      └──────────────────┘     └──────────────────┘
```

---

## 🔧 Solução 1: Conv-Adapter

### **Arquitetura Detalhada**

```
┌────────────────────────────────────────────────────────────┐
│                       STAGE 1                              │
│  ┌──────────────┐         ┌──────────────┐               │
│  │  Input Block │────────▶│   Backbone   │──────┐        │
│  │  [1,16,16]   │         │  (ResNet-18) │      │        │
│  └──────────────┘         └──────────────┘      │        │
│                                                  │        │
│                                           ┌──────▼──────┐ │
│                                           │ Binary Head │ │
│                                           │ NONE vs     │ │
│                                           │ PARTITION   │ │
│                                           └─────────────┘ │
│  TREINA NORMAL (end-to-end)                              │
└────────────────────────────────────────────────────────────┘
                         │
                         │ CONGELA BACKBONE
                         ▼
┌────────────────────────────────────────────────────────────┐
│                       STAGE 2                              │
│  ┌──────────────┐         ┌──────────────┐               │
│  │  Input Block │────────▶│   Backbone   │──────┐        │
│  │  [1,16,16]   │         │  (FROZEN)    │      │        │
│  └──────────────┘         └──────────────┘      │        │
│                                                  │        │
│                           ┌──────────────────────▼────┐   │
│                           │    Conv-Adapter          │   │
│                           │  ┌────────────────────┐  │   │
│                           │  │ Down: 512→128      │  │   │
│                           │  │ (point-wise)       │  │   │
│                           │  ├────────────────────┤  │   │
│                           │  │ DW-Conv 3x3        │  │   │
│                           │  │ (maintains locality)│  │   │
│                           │  ├────────────────────┤  │   │
│                           │  │ Up: 128→512        │  │   │
│                           │  │ (point-wise)       │  │   │
│                           │  └────────────────────┘  │   │
│                           │                          │   │
│                           │  h' = h + α·Δh          │   │
│                           └──────────────────────────┘   │
│                                                  │        │
│                                           ┌──────▼──────┐ │
│                                           │   3-way Head│ │
│                                           │ SPLIT, RECT,│ │
│                                           │    AB       │ │
│                                           └─────────────┘ │
│  TREINA APENAS: Adapter (3.5% params) + Head             │
└────────────────────────────────────────────────────────────┘
```

### **Formulação Matemática**

```
h_frozen = Backbone(x)  [512-dim, frozen]
Δh = Up(ReLU(DW-Conv(ReLU(Down(h_frozen)))))
h_adapted = h_frozen + α ⊙ Δh  [element-wise]
logits = Head(h_adapted)
```

**Parâmetros:**
- Down projection: 512×128 = 65K
- DW-Conv 3×3: 128×9 = 1.2K
- Up projection: 128×512 = 65K
- α (learnable): 512
- **Total adapter: ~131K params (vs 11M backbone = 1.2%)**

---

## 🎲 Solução 2: Multi-Stage Ensemble

### **Arquitetura Detalhada**

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE DIVERSO                             │
│                                                                 │
│  Input Block [1,16,16]                                         │
│        │                                                        │
│        ├──────────────┬──────────────┬──────────────┐          │
│        │              │              │              │          │
│        ▼              ▼              ▼              │          │
│  ┌─────────┐   ┌──────────┐   ┌───────────┐       │          │
│  │ResNet-18│   │MobileNet │   │EfficientNet│       │          │
│  │Pipeline │   │Pipeline  │   │ Pipeline  │       │          │
│  └────┬────┘   └────┬─────┘   └─────┬─────┘       │          │
│       │             │               │              │          │
│       │  Stage 1: Binary (NONE vs PARTITION)      │          │
│       │             │               │              │          │
│       ▼             ▼               ▼              │          │
│  ┌─────────────────────────────────────┐           │          │
│  │      Soft Voting (learnable α)     │           │          │
│  │  P_ensemble = Σ(α_i · P_i)        │           │          │
│  └──────────────┬──────────────────────┘           │          │
│                 │                                  │          │
│                 ▼                                  │          │
│  ┌─────────────────────────────────────┐           │          │
│  │      Stage 2: 3-way (per model)    │           │          │
│  └──────────────┬──────────────────────┘           │          │
│                 │                                  │          │
│                 ▼                                  │          │
│  ┌─────────────────────────────────────┐           │          │
│  │      Soft Voting (stage 2)         │           │          │
│  └──────────────┬──────────────────────┘           │          │
│                 │                                  │          │
│                 ▼                                  │          │
│  ┌─────────────────────────────────────┐           │          │
│  │   Stage 3-RECT / 3-AB (per model)  │           │          │
│  └──────────────┬──────────────────────┘           │          │
│                 │                                  │          │
│                 ▼                                  │          │
│  ┌─────────────────────────────────────┐           │          │
│  │   Final Prediction (0-9)           │           │          │
│  └─────────────────────────────────────┘           │          │
└─────────────────────────────────────────────────────────────────┘
```

**Diversidade:**
1. ResNet-18: Deep residual (11M params)
2. MobileNetV2: Efficient inverted residuals (3.5M params)
3. EfficientNet-B0: Compound scaling (5M params)

**Voting:**
```python
P_ensemble = softmax(α) · [P_model1, P_model2, P_model3]^T
α = learnable weights (inicializado: [1/3, 1/3, 1/3])
```

---

## ⭐ Solução 3: HÍBRIDO (Conv-Adapter + Ensemble)

### **Arquitetura Detalhada**

```
┌─────────────────────────────────────────────────────────────────┐
│                   SOLUÇÃO HÍBRIDA (NOVEL)                       │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              STAGE 1 (Pre-trained)                     │    │
│  │  Backbone + Binary Head                                │    │
│  │  (treinado normalmente, depois CONGELADO)             │    │
│  └────────────────────────────────────────────────────────┘    │
│                         │                                       │
│                         │ Backbone CONGELADO                    │
│                         ▼                                       │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              STAGE 2 (Adapter Ensemble)                │    │
│  │                                                         │    │
│  │  Input Block                                           │    │
│  │        │                                                │    │
│  │        └──────────┬──────────┬──────────┐             │    │
│  │                   │          │          │             │    │
│  │            ┌──────▼────┐ ┌──▼─────┐ ┌──▼─────┐       │    │
│  │            │ Adapter 1 │ │Adapter2│ │Adapter3│       │    │
│  │            │ (γ=4)     │ │(γ=8)   │ │(γ=4)   │       │    │
│  │            └─────┬─────┘ └───┬────┘ └───┬────┘       │    │
│  │                  │           │          │             │    │
│  │            ┌─────▼───┐  ┌────▼───┐ ┌────▼───┐        │    │
│  │            │ Head 1  │  │ Head 2 │ │ Head 3 │        │    │
│  │            └─────┬───┘  └────┬───┘ └────┬───┘        │    │
│  │                  │           │          │             │    │
│  │                  └───────────┼──────────┘             │    │
│  │                              │                        │    │
│  │                    ┌─────────▼────────┐               │    │
│  │                    │  Soft Voting     │               │    │
│  │                    │  (learnable α)   │               │    │
│  │                    └──────────────────┘               │    │
│  │                                                        │    │
│  │  PARAMS: ~10% of full fine-tuning (3 adapters × 3.5%)│    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Similar structure for Stage 3-RECT and Stage 3-AB             │
└─────────────────────────────────────────────────────────────────┘
```

### **Configurações dos Adapters (Diversidade)**

```python
Adapter 1:
  - reduction: 4 (512→128)
  - layers: ['layer3', 'layer4']
  - variant: 'conv_parallel'

Adapter 2:
  - reduction: 8 (512→64)  # Mais compacto
  - layers: ['layer3', 'layer4']
  - variant: 'conv_parallel'

Adapter 3:
  - reduction: 4 (512→128)
  - layers: ['layer4']  # Apenas camada profunda
  - variant: 'conv_parallel'
```

**Diversidade alcançada por:**
1. Diferentes reduction ratios (4 vs 8)
2. Diferentes layers adaptadas (layer3+4 vs só layer4)
3. Inicializações randômicas diferentes

### **Forward Pass Matemático**

```python
# Backbone congelado
h_frozen = Backbone(x)  # [B, 512]

# 3 adapters modulam de formas diferentes
h1 = h_frozen + α1 ⊙ Adapter1(h_frozen)
h2 = h_frozen + α2 ⊙ Adapter2(h_frozen)
h3 = h_frozen + α3 ⊙ Adapter3(h_frozen)

# 3 heads classificam
logits1 = Head1(h1)  # [B, num_classes]
logits2 = Head2(h2)
logits3 = Head3(h3)

# Soft voting
P1, P2, P3 = softmax(logits1), softmax(logits2), softmax(logits3)
weights = softmax([w1, w2, w3])  # learnable
P_ensemble = weights[0]·P1 + weights[1]·P2 + weights[2]·P3

final_logits = log(P_ensemble)
```

---

## 📊 Comparação das Soluções

| Aspecto | Baseline v6 | Conv-Adapter | Ensemble | HÍBRIDO |
|---------|------------|--------------|----------|---------|
| **Negative Transfer** | ✗ Sim (46%→32%) | ✓ Resolvido | ✓ Mitigado | ✓ Resolvido |
| **Trainable Params** | 100% (11M) | 3.5% (385K) | 300% (33M) | ~10% (1.1M) |
| **Training Time** | 1× | 0.3× | 3× | 0.9× |
| **Inference Speed** | 1× | 1× | 0.33× | 0.8× |
| **Few-shot Robustness** | Baixa | Alta | Média | Alta |
| **Expected Stage2 F1** | 46% | 60-65% | 50-55% | **65-73%** |
| **Expected Stage3-AB F1** | 24.5% | 40-45% | 30-35% | **45-50%** |
| **Expected Pipeline F1** | ~65% | 68-72% | ~70% | **70-75%** |
| **Inovação PhD** | - | Aplicação | Aplicação | **NOVEL** |

---

## 🎓 Contribuições Científicas

### **Solução 1: Conv-Adapter**
- **Tipo:** Aplicação de método SOTA (Chen et al., CVPR 2024)
- **Contribuição:** Primeira aplicação em predição de partições de codec
- **Novidade:** Domínio (CV → Video Compression)

### **Solução 2: Ensemble**
- **Tipo:** Aplicação de método estabelecido (Ahad et al., 2024)
- **Contribuição:** Ensemble hierárquico multi-stage (não apenas final)
- **Novidade:** Voting em CADA estágio da hierarquia

### **Solução 3: HÍBRIDO** ⭐
- **Tipo:** **Combinação INÉDITA na literatura**
- **Contribuição:** Adapter + Ensemble para efficiency + robustness
- **Novidade:** 
  - Trade-off otimizado (10% params, 70-75% F1)
  - Ensemble de adapters (não de backbones)
  - Aplicação a classificação hierárquica desbalanceada
- **Potencial publicação:** Workshop/Journal paper

---

## 🔬 Protocolo de Validação

### **Ablation Studies**

1. **Conv-Adapter:**
   - [ ] γ (reduction): 2, 4, 8, 16
   - [ ] Layers: layer4, layer3+4, all layers
   - [ ] Variant: conv_parallel vs residual_parallel

2. **Ensemble:**
   - [ ] Num models: 2, 3, 5
   - [ ] Voting: soft vs hard
   - [ ] Weights: fixed vs learnable

3. **Híbrido:**
   - [ ] Num adapters: 2, 3, 5
   - [ ] Adapter configs: diversidade vs uniformidade
   - [ ] Voting weights: inicialização

### **Controles Experimentais**

- **Dataset:** Mesmo split (train/val) para TODAS soluções
- **Seed:** 42 (reprodutibilidade)
- **Hardware:** Mesma GPU (RTX 3090)
- **Hiperparâmetros base:** lr=1e-4, batch=128, epochs=100

### **Métricas de Comparação**

✅ **Primárias:**
- F1-score macro (stage-wise e pipeline)
- F1 per-class (especialmente classes raras)
- Confusion matrix

✅ **Secundárias:**
- Parameter efficiency (% trainable)
- Training time (epochs to convergence)
- Inference speed (samples/sec)
- GPU memory usage

---

## 📖 Documentação Requerida

Para cada solução, criar em `docs_v7/`:

1. **Motivação** (problema específico que resolve)
2. **Literatura** (5-10 papers citados)
3. **Hipótese** (quantitativa, testável)
4. **Protocolo** (reprodutível passo-a-passo)
5. **Arquitetura** (diagramas + equações)
6. **Resultados** (tabelas + gráficos)
7. **Análise** (por que funcionou/falhou?)
8. **Limitações** (o que não testamos?)
9. **Contribuições** (inovação acadêmica)
10. **Referências** (bibliografia completa)

---

**Autor:** Chiaro Rosa  
**Data:** 14 de Outubro de 2025  
**Versão:** 7.0.0
