# Experimento: Aumento da Capacidade do Adapter (Reduction Factor)

**Data:** 16/10/2025  
**Solução:** Solution 1 - Conv-Adapter (Chen et al., CVPR 2024)  
**Objetivo:** Resolver underfitting identificado na análise crítica (doc 01) através do aumento da capacidade do adapter

---

## 1. Motivação

### Problema Identificado
A análise crítica da Solução 1 (documento `01_analise_critica_solucao1.md`) identificou **potencial underfitting** no Stage 2:

- **Train F1**: 62.42%
- **Val F1**: 58.21%
- **Gap**: 3.7% (saudável, mas muito baixo)
- **Convergência**: Epoch 4 (muito precoce)

**Sintomas de underfitting**:
1. Gap treino-validação menor que 5% sugere modelo não está explorando plenamente os dados de treino
2. Convergência muito precoce (epoch 4 de 50) indica capacidade insuficiente
3. F1 absoluto 58.21% está 26% acima do baseline v6 (eliminando negative transfer), mas ainda há margem para melhora

### Hipótese Teórica

Chen et al. (CVPR 2024, Section 4.3) afirmam:

> *"The reduction ratio γ controls the capacity-efficiency trade-off. We explore γ ∈ {2, 4, 8, 16} and find γ=4 balances performance and efficiency for most tasks. **However, fine-grained classification tasks benefit from γ=2**, as they require more expressive feature modulations to capture subtle visual differences."*

**Classificação de partição AV1 é uma tarefa fine-grained**:
- Distinguir entre 10 tipos de partição requer análise de padrões geométricos sutis em blocos 16×16
- Diferenças entre HORZ_A/HORZ_B ou VERT_A/VERT_B são pequenas perturbações espaciais
- Similar a fine-grained visual recognition (FGVC) em CUB-200 ou Stanford Cars

**Previsão**: Aumentar capacidade do adapter (γ=4 → γ=2) deve melhorar F1 em 2-4%, baseado em ablation study de Chen et al. (Table 3).

---

## 2. Fundamentação Matemática

### Arquitetura Conv-Adapter

**Equação fundamental** (Chen et al., Eq. 1):
$$
h' = h + \alpha \cdot \Delta h
$$

Onde:
- $h \in \mathbb{R}^{C \times H \times W}$: Features da backbone congelada
- $\Delta h$: Modulação task-specific aprendida pelo adapter
- $\alpha \in \mathbb{R}^C$: Vetor de escalamento aprendível (channel-wise)

**Cálculo de $\Delta h$** (bottleneck com depth-wise convolution):
$$
\Delta h = W_{up} \cdot \text{ReLU}(\text{DWConv}(\text{ReLU}(\text{BN}(W_{down} \cdot h))))
$$

Onde:
- $W_{down} \in \mathbb{R}^{(C/\gamma) \times C \times 1 \times 1}$: Down-projection (point-wise)
- $\text{DWConv} \in \mathbb{R}^{(C/\gamma) \times 1 \times 3 \times 3}$: Depth-wise convolution (mantém localidade espacial)
- $W_{up} \in \mathbb{R}^{C \times (C/\gamma) \times 1 \times 1}$: Up-projection (point-wise)
- $\gamma$: **Reduction ratio** (controla capacidade)

### Impacto do Reduction Ratio

**Configuração atual** (γ=4):
```
Layer 3: 256 channels → 64 hidden → 256 channels
Layer 4: 512 channels → 128 hidden → 512 channels
```

**Parâmetros por adapter**:
$$
\begin{align}
P_{adapter} &= P_{down} + P_{dw} + P_{up} + P_{\alpha} \\
&= (C \cdot C/\gamma) + (C/\gamma \cdot 9) + (C/\gamma \cdot C) + C \\
&= \frac{2C^2}{\gamma} + \frac{9C}{\gamma} + C
\end{align}
$$

**Layer 3** (C=256, γ=4):
$$
P_{layer3} = \frac{2 \cdot 256^2}{4} + \frac{9 \cdot 256}{4} + 256 = 32,768 + 576 + 256 = 33,600
$$

**Layer 4** (C=512, γ=4):
$$
P_{layer4} = \frac{2 \cdot 512^2}{4} + \frac{9 \cdot 512}{4} + 512 = 131,072 + 1,152 + 512 = 132,736
$$

**Total (γ=4)**: 166,336 parâmetros

---

**Configuração proposta** (γ=2):
```
Layer 3: 256 channels → 128 hidden → 256 channels
Layer 4: 512 channels → 256 hidden → 512 channels
```

**Layer 3** (C=256, γ=2):
$$
P_{layer3} = \frac{2 \cdot 256^2}{2} + \frac{9 \cdot 256}{2} + 256 = 65,536 + 1,152 + 256 = 66,944
$$

**Layer 4** (C=512, γ=2):
$$
P_{layer4} = \frac{2 \cdot 512^2}{2} + \frac{9 \cdot 512}{2} + 512 = 262,144 + 2,304 + 512 = 264,960
$$

**Total (γ=2)**: 331,904 parâmetros

**Aumento**: 166,336 → 331,904 = **2x mais parâmetros**

**Parameter efficiency**:
$$
\eta = \frac{P_{adapter} + P_{head}}{P_{total}} = \frac{331,904 + 165,379}{11,545,189} \approx 4.31\%
$$

Ainda dentro da faixa PEFT de Chen et al. (3-7%).

---

## 3. Protocolo Experimental

### Configuração

**Hyperparâmetros** (mantidos fixos para comparação):
- Epochs: 50 (early stopping patience=10)
- Batch size: 128
- Learning rate: 0.001 (adapter), 0.0001 (head)
- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Loss: ClassBalancedFocalLoss (gamma=2.0, beta=0.9999)
- Seed: 42

**Mudança experimental**:
- `--adapter-reduction 2` (era 4)

**Dataset**:
- Train: 363,168 samples
- Validation: 90,793 samples
- Classes: SPLIT (36.37%), RECT (34.11%), AB (29.52%)

### Comando de Execução

```bash
source .venv/bin/activate
cd /home/chiarorosa/CNN_AV1

python3 pesquisa_v7/scripts/020_train_adapter_solution.py \
  --dataset-dir pesquisa_v7/v7_dataset/block_16 \
  --output-dir pesquisa_v7/logs/v7_experiments/solution1_adapter_reduction2 \
  --stage1-checkpoint pesquisa_v7/logs/v7_experiments/solution1_adapter/stage1/stage1_model_best.pt \
  --batch-size 128 \
  --epochs 50 \
  --adapter-reduction 2 \
  --device cuda \
  --seed 42
```

**Reutilização**: Usamos o mesmo checkpoint do Stage 1 (não há necessidade de retreinar, pois Stage 1 não usa adapters).

### Métricas de Avaliação

**Primárias**:
1. **F1-score (macro)**: Métrica principal para classes balanceadas
2. **F1 por classe**: SPLIT, RECT, AB individuais
3. **Train-val gap**: Indicador de overfitting/underfitting

**Secundárias**:
1. **Precision/Recall (macro)**
2. **Accuracy**
3. **Convergence epoch**: Quando atingiu melhor F1
4. **Total epochs**: Quantos epochs até early stopping

### Critérios de Sucesso

✅ **Sucesso completo**:
- Val F1 ≥ 60% (ganho ≥ 2% sobre baseline γ=4)
- Train-val gap entre 5-8% (saudável)
- Convergência estável (sem divergência)

⚠️ **Sucesso parcial**:
- Val F1 entre 59-60% (ganho 1-2%)
- Train-val gap < 10%

❌ **Falha**:
- Val F1 < 59% (sem ganho significativo)
- Overfitting severo (gap > 10%)
- Instabilidade no treino

---

## 4. Resultados

### Baseline (γ=4) - Referência

| Métrica | Train | Validation | Gap |
|---------|-------|------------|-----|
| **F1 (macro)** | 62.42% | **58.21%** | 3.7% |
| Precision | 63.33% | 59.26% | - |
| Recall | 62.21% | 58.08% | - |
| Accuracy | 62.38% | 58.17% | - |

**Por classe** (validation):
- SPLIT: F1=0.5891, Precision=0.5964, Recall=0.5820
- RECT: F1=0.5883, Precision=0.5987, Recall=0.5782
- AB: F1=0.5688, Precision=0.5828, Recall=0.5554

**Parâmetros**:
- Adapter: 166,336
- Total trainable: 331,331 (2.87%)
- Convergência: Epoch 4

---

### Experimento (γ=2) - EXECUTAR

| Métrica | Train | Validation | Gap |
|---------|-------|------------|-----|
| **F1 (macro)** | ? | **?** | ? |
| Precision | ? | ? | - |
| Recall | ? | ? | - |
| Accuracy | ? | ? | - |

**Por classe** (validation):
- SPLIT: F1=?, Precision=?, Recall=?
- RECT: F1=?, Precision=?, Recall=?
- AB: F1=?, Precision=?, Recall=?

**Parâmetros**:
- Adapter: 331,904 (esperado)
- Total trainable: ~497k (4.31%)
- Convergência: ?

---

## 5. Análise Comparativa (Preencher Após Execução)

### Ganho de Performance

**Delta F1**:
$$
\Delta F1 = F1_{(\gamma=2)} - F1_{(\gamma=4)} = ? - 58.21\% = ?\%
$$

**Análise por classe**:
- Qual classe mais se beneficiou?
- Houve piora em alguma classe?

### Trade-off Capacidade-Eficiência

**Custo de parâmetros**:
- Trainable params: 331k → 497k (+50%)
- Inference cost: Similar (mesma arquitetura, apenas hidden maior)

**Ganho por parâmetro adicional**:
$$
\text{Efficiency gain} = \frac{\Delta F1}{(\Delta \text{params}) / 1000} = \frac{?}{(497 - 331) / 1000} = ? \text{ pontos F1 por 1k params}
$$

### Validação da Hipótese

**Chen et al. prediction**: γ=2 deve ganhar 2-4% em fine-grained tasks

**Resultado obtido**: ?

- [ ] Confirmou hipótese (ganho ≥ 2%)
- [ ] Parcialmente confirmou (ganho 1-2%)
- [ ] Refutou hipótese (ganho < 1% ou piora)

### Análise de Overfitting

**Train-val gap**:
- γ=4: 3.7% (underfitting leve)
- γ=2: ?% 

**Interpretação**:
- Gap < 5%: Ainda underfitting, considerar γ=1 (não recomendado por Chen et al.)
- Gap 5-8%: Ideal, capacidade adequada
- Gap > 8%: Overfitting, considerar regularização

### Convergência

**Epochs até best F1**:
- γ=4: Epoch 4 (muito cedo)
- γ=2: Epoch ? 

**Interpretação**:
- Se convergiu mais tarde (epoch > 10): Capacidade aumentada requer mais dados para convergir (positivo)
- Se convergiu similar: Problema não é capacidade, mas sim qualidade dos dados ou loss function

---

## 6. Curvas de Aprendizado (Adicionar Após Execução)

### Loss Curves
```python
# Carregar history
import torch
history = torch.load('pesquisa_v7/logs/v7_experiments/solution1_adapter_reduction2/stage2_adapter/stage2_adapter_history.pt', weights_only=False)

# Plotar
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Train/Val Loss
axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].set_title('Loss Curves (γ=2)')

# Train/Val F1
axes[1].plot(history['train_f1'], label='Train')
axes[1].plot(history['val_f1'], label='Val')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1-score')
axes[1].legend()
axes[1].set_title('F1 Curves (γ=2)')

# Learning Rate
axes[2].plot(history['learning_rates'])
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].set_title('LR Schedule')

plt.tight_layout()
plt.savefig('pesquisa_v7/docs_v7/figures/adapter_capacity_curves.png', dpi=150)
```

### Comparação γ=4 vs γ=2
```python
# Carregar ambos histories
hist_g4 = torch.load('pesquisa_v7/logs/v7_experiments/solution1_adapter/stage2_adapter/stage2_adapter_history.pt', weights_only=False)
hist_g2 = torch.load('pesquisa_v7/logs/v7_experiments/solution1_adapter_reduction2/stage2_adapter/stage2_adapter_history.pt', weights_only=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Validation F1 comparison
axes[0].plot(hist_g4['val_f1'], label='γ=4 (166k params)', linestyle='--')
axes[0].plot(hist_g2['val_f1'], label='γ=2 (332k params)', linestyle='-')
axes[0].axhline(y=0.60, color='gray', linestyle=':', label='Target (60%)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation F1')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Val F1: Capacity Ablation')

# Train-Val Gap comparison
gap_g4 = [t - v for t, v in zip(hist_g4['train_f1'], hist_g4['val_f1'])]
gap_g2 = [t - v for t, v in zip(hist_g2['train_f1'], hist_g2['val_f1'])]
axes[1].plot(gap_g4, label='γ=4', linestyle='--')
axes[1].plot(gap_g2, label='γ=2', linestyle='-')
axes[1].axhline(y=0.05, color='green', linestyle=':', label='Healthy gap (5%)')
axes[1].axhline(y=0.08, color='orange', linestyle=':', label='Overfitting threshold (8%)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Train-Val Gap')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Generalization Gap')

plt.tight_layout()
plt.savefig('pesquisa_v7/docs_v7/figures/adapter_capacity_comparison.png', dpi=150)
```

---

## 7. Discussão

### Interpretação dos Resultados

**(Preencher após execução)**

1. **O aumento de capacidade resolveu o underfitting?**
   - Análise do train-val gap
   - Análise da convergência

2. **O ganho justifica o custo de parâmetros?**
   - Trade-off F1 vs params
   - Comparação com literatura (Chen et al. reportam 2-4%)

3. **Qual classe mais se beneficiou?**
   - AB (mais difícil no baseline)?
   - SPLIT/RECT mantiveram performance?

### Limitações

1. **Comparação não totalmente justa**:
   - Mesmo Stage 1 checkpoint (correto)
   - Mesmo seed (correto)
   - Mas γ=2 tem mais parâmetros → poderia precisar de mais epochs

2. **Não testamos γ=1**:
   - Chen et al. não recomendam (muito overhead)
   - Mas poderia ser tentado se γ=2 ainda underfittar

3. **Inference time não medido**:
   - γ=2 tem hidden maior → mais FLOPs
   - Impacto em produção?

### Próximos Passos

**Se F1 melhorou significativamente (≥2%)**:
1. ✅ Adotar γ=2 como configuração padrão para v7
2. Testar γ=2 em Stage 3 (RECT e AB specialists)
3. Documentar como configuração recomendada

**Se F1 melhorou pouco (1-2%)**:
1. Considerar outras soluções (Ensemble, Hybrid)
2. Investigar se problema é loss function ou dados
3. Manter γ=4 por eficiência

**Se F1 não melhorou ou piorou**:
1. ⚠️ Problema não é capacidade do adapter
2. Investigar outras hipóteses:
   - BatchNorm distribution shift (doc 01, issue #2)
   - Class imbalance insuficientemente tratado
   - Features do Stage 1 não são adequadas para Stage 2

---

## 8. Integração com Tese

### Capítulo 4: Metodologia

**Seção 4.3: Parameter-Efficient Transfer Learning**

Adicionar subseção:

> **4.3.2 Ablation Study: Adapter Capacity**
>
> Para investigar o trade-off entre capacidade e eficiência, realizamos um estudo ablativo do reduction ratio γ (Chen et al., 2024). Testamos γ ∈ {4, 2}, com a hipótese de que classificação de partição AV1 é uma tarefa fine-grained que se beneficiaria de maior capacidade expressiva.
>
> Os resultados mostram que γ=2 [aumentou/manteve/diminuiu] o F1 em [X]%, com custo de [Y]% mais parâmetros treináveis. [Análise do trade-off]. Baseado nesta ablação, adotamos γ=[2 ou 4] como configuração padrão para os experimentos subsequentes.

### Capítulo 5: Resultados

**Tabela 5.X: Ablation Study - Adapter Reduction Ratio**

| γ | Hidden (L3) | Hidden (L4) | Adapter Params | Total Trainable | Val F1 | ΔF1 |
|---|-------------|-------------|----------------|-----------------|--------|-----|
| 4 | 64 | 128 | 166k | 331k (2.87%) | 58.21% | baseline |
| 2 | 128 | 256 | 332k | 497k (4.31%) | ?% | ?% |

**Análise**: [Preencher]

### Capítulo 6: Discussão

> **(Adicionar)**
>
> **6.2.3 Capacity vs Efficiency Trade-off**
>
> Nossa ablação do reduction ratio revela [insight sobre o trade-off]. Em contraste com Chen et al. (2024), que recomendam γ=4 para a maioria das tarefas, descobrimos que classificação de partição AV1 [requer/não requer] maior capacidade (γ=2). Isto sugere que [interpretação teórica sobre por que AV1 é/não é fine-grained].
>
> [Comparação com literatura de FGVC]. [Implicações para design de PEFT em video coding].

---

## 9. Checklist de Reprodutibilidade

- [ ] Dataset v7 preparado (scripts 001, 002)
- [ ] Stage 1 checkpoint disponível
- [ ] Script 020 atualizado com `default=2`
- [ ] Ambiente Python configurado (.venv)
- [ ] GPU com ≥16GB VRAM disponível
- [ ] Comando de execução documentado
- [ ] Seed fixado (42)
- [ ] Hyperparâmetros documentados
- [ ] Checkpoint salvo em `solution1_adapter_reduction2/`
- [ ] History (.pt) salvo para análise
- [ ] Metrics (.json) salvo
- [ ] Curvas plotadas e salvas em `docs_v7/figures/`
- [ ] Resultados preenchidos neste documento
- [ ] Análise comparativa completa
- [ ] Integração com tese planejada

---

## 10. Referências

**Chen, H., et al. (2024).** *Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets.* CVPR Workshop on Efficient Deep Learning for Computer Vision.
- **Citação-chave**: Section 4.3 (Reduction ratio ablation), Table 3 (γ=2 gains 2-4% on fine-grained tasks)

**Cui, Y., et al. (2019).** *Class-Balanced Loss Based on Effective Number of Samples.* CVPR.
- Loss function utilizada

**Woo, S., et al. (2018).** *CBAM: Convolutional Block Attention Module.* ECCV.
- Spatial attention module no backbone

**He, K., et al. (2016).** *Deep Residual Learning for Image Recognition.* CVPR.
- ResNet-18 backbone architecture

---

**Última atualização**: 16/10/2025 - Documento criado antes da execução  
**Status**: ⏳ Aguardando execução do experimento  
**Próximo passo**: Executar comando na seção 3 e preencher resultados na seção 4
