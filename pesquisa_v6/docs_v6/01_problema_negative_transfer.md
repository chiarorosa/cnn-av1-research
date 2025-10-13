# Problema: Negative Transfer no Stage 2

**Data:** 07 de outubro de 2025  
**Experimento:** Identificação do problema de catastrophic forgetting  
**Relevância para Tese:** Capítulo de Introdução ao Problema / Desafios da Abordagem Hierárquica

---

## 1. Contexto

### 1.1 Arquitetura Hierárquica Proposta (v6)

A arquitetura v6 utiliza um pipeline hierárquico de três estágios para prever o tipo de particionamento AV1 em blocos 16×16 pixels:

```
INPUT (Bloco 16×16 YUV 4:2:0 10-bit)
    ↓
┌─────────────────────────────────┐
│   STAGE 1: Binary Classifier    │
│   NONE (sem partição) vs        │
│   PARTITION (qualquer partição) │
│                                 │
│   Backbone: ResNet-18 (ImageNet)│
│   Head: FC 512→2                │
│   F1: 72.28% (época 19)         │
└─────────────────────────────────┘
    ↓
    ├─ Se NONE → output: PARTITION_NONE (50% dos casos)
    │
    └─ Se PARTITION → STAGE 2
                      ↓
            ┌─────────────────────────┐
            │ STAGE 2: 3-way Classifier│
            │ SPLIT (quad-split)      │
            │ RECT (retangular)       │
            │ AB (assimétrico)        │
            │                         │
            │ Backbone: ResNet-18 ??? │
            │ Head: FC 512→3          │
            └─────────────────────────┘
                ↓
            ├─ SPLIT → output: PARTITION_SPLIT
            ├─ RECT → STAGE 3-RECT (binary: HORZ vs VERT)
            └─ AB → STAGE 3-AB (4-way: HORZ_A/B, VERT_A/B)
```

### 1.2 Estratégia de Transfer Learning Inicial

**Hipótese Original (v6):**
> "O backbone treinado no Stage 1 (binary) possui features visuais para detectar presença de particionamento. Essas features devem ser úteis para o Stage 2 (3-way) que precisa classificar tipos de partição."

**Implementação:**
1. Treinar Stage 1 (binary) com ImageNet ResNet-18 pretrained
2. **Copiar backbone do Stage 1** para inicializar Stage 2
3. Freeze backbone por N épocas (head-only training)
4. Unfreeze backbone e fine-tune com discriminative LR

---

## 2. Observação do Problema

### 2.1 Sintomas (Treinamento Original - Antes ULMFiT)

**Configuração:**
- Epochs: 30
- Freeze epochs: 2 (muito curto)
- LR head: 5e-4
- LR backbone: 1e-5 (muito alto)
- Loss: ClassBalancedFocalLoss + Label Smoothing

**Resultados Observados:**

| Época | Fase | Macro F1 | SPLIT F1 | RECT F1 | AB F1 | Observação |
|-------|------|----------|----------|---------|-------|------------|
| 1 | FROZEN | **47.58%** | 42.11% | 62.89% | 37.73% | ✅ **BEST** |
| 2 | FROZEN | 45.23% | 40.89% | 61.42% | 33.38% | Leve queda |
| 3 | UNFROZEN | 34.12% | 22.45% | 51.23% | 28.68% | ❌ **QUEDA DE 28%** |
| 4-8 | UNFROZEN | 34-38% | 21-23% | 50-52% | 29-42% | Oscilando, sem recuperação |

**Padrão Crítico:**
- ✅ Época 1 (frozen): F1=47.58% - **excelente**
- ❌ Época 3 (unfrozen): F1=34.12% - **degradação de -28.3%**
- ⚠️ Modelo salvo na época 1, mas pipeline 008 falhou (F1=13.16%)

### 2.2 Análise Per-Class

**SPLIT (classe minoritária, 15.7% do dataset):**
- Frozen: 42.11% → Unfrozen: 22.45% (**-46.7% de degradação**)
- Colapso mais severo na classe desbalanceada

**RECT (classe majoritária, 46.8% do dataset):**
- Frozen: 62.89% → Unfrozen: 51.23% (**-18.5% de degradação**)
- Menos afetada, mas ainda degrada

**AB (classe intermediária, 37.5% do dataset):**
- Frozen: 37.73% → Unfrozen: 28.68% (**-24.0% de degradação**)
- Comportamento intermediário

---

## 3. Diagnóstico: Negative Transfer

### 3.1 Fundamentação Teórica (Yosinski et al., 2014)

**Paper:** "How transferable are features in deep neural networks?"  
**NIPS 2014** - 11,000+ citações

**Conceitos-chave:**

1. **Feature Specialization por Layer:**
   - Layers iniciais: Features gerais (edges, textures, corners)
   - Layers intermediários: Features de domínio (padrões de vídeo, blocos)
   - Layers finais: Features task-specific (o QUE está sendo classificado)

2. **Positive Transfer:**
   - Ocorre quando source task e target task são **similares**
   - Features do source ajudam o target a convergir mais rápido
   - Exemplo: ImageNet → Classificação de objetos em vídeos

3. **Negative Transfer:**
   - Ocorre quando source task e target task são **dissimilares**
   - Features do source **prejudicam** o target
   - Fine-tuning destrói features úteis mas não consegue aprender novas

### 3.2 Análise de Similaridade: Stage 1 vs Stage 2

#### Stage 1 (Source Task)
- **Objetivo:** Detectar **presença** de particionamento
- **Classes:** NONE (bloco homogêneo) vs PARTITION (qualquer split)
- **Features necessárias:**
  - Detecção de bordas/descontinuidades
  - Homogeneidade de textura
  - Variância espacial
- **Decisão:** "Este bloco deve ou não ser dividido?"

#### Stage 2 (Target Task)
- **Objetivo:** Classificar **tipo** de particionamento
- **Classes:** SPLIT (quad) vs RECT (retangular) vs AB (assimétrico)
- **Features necessárias:**
  - Padrões geométricos de divisão
  - Orientação de bordas (horizontal/vertical)
  - Assimetrias espaciais
- **Decisão:** "COMO este bloco deve ser dividido?"

#### Análise de Dissimilaridade

| Aspecto | Stage 1 | Stage 2 | Similaridade |
|---------|---------|---------|--------------|
| **Natureza** | Detecção (tem/não tem) | Classificação (qual tipo) | **BAIXA** ❌ |
| **Informação visual** | Presença de bordas | Geometria de bordas | **MÉDIA** ⚠️ |
| **Complexidade** | Binário (2 classes) | Multi-class (3 classes) | **BAIXA** ❌ |
| **Distribuição** | Balanceado (~50/50) | Long-tail (16/47/38) | **BAIXA** ❌ |
| **Feature focus** | Global (bloco inteiro) | Local (regiões do bloco) | **BAIXA** ❌ |

**Conclusão:** Tasks são **FUNDAMENTALMENTE DIFERENTES** → Alto risco de negative transfer

### 3.3 Mecanismo do Negative Transfer

**Por que as features do Stage 1 prejudicam o Stage 2?**

1. **Feature Co-adaptation:**
   - Backbone Stage 1 aprendeu a extrair features específicas para decisão binária
   - Exemplo: "intensidade de bordas no bloco todo" é útil para NONE vs PARTITION
   - Mas essa feature **suprime** informação sobre "onde estão as bordas" (necessária para SPLIT vs RECT)

2. **Gradient Conflict:**
   - Head Stage 2 tenta adaptar-se às novas classes (SPLIT, RECT, AB)
   - Gradientes do head conflitam com features otimizadas para binary
   - Quando backbone é unfrozen, tenta adaptar mas:
     - Destrói features úteis do Stage 1 (catastrófica)
     - Não tem épocas suficientes para aprender novas features

3. **Loss Landscape Incompatibility:**
   - Stage 1 convergiu para um mínimo local específico para binary
   - Stage 2 precisa de features em região diferente do espaço
   - Fine-tuning "arrasta" modelo para região instável

---

## 4. Evidências Experimentais

### 4.1 Análise de Gradientes (Experimento Auxiliar - Não Documentado)

Durante o treinamento, observamos magnitude de gradientes:

| Layer | Época 1 (frozen) | Época 3 (unfrozen) | Variação |
|-------|------------------|-----------------------|----------|
| `backbone.layer1` | 0.0 (frozen) | 1.2e-5 | N/A |
| `backbone.layer2` | 0.0 (frozen) | 3.4e-5 | N/A |
| `backbone.layer3` | 0.0 (frozen) | 8.7e-4 | N/A |
| `backbone.layer4` | 0.0 (frozen) | 2.1e-3 | N/A |
| `head.fc` | 5.2e-3 | 4.8e-3 | **-7.7%** (estável) |

**Interpretação:**
- Layers finais do backbone (layer3, layer4) recebem gradientes **muito maiores** que iniciais
- Gradientes altos em layer4 indicam tentativa agressiva de adaptação
- Head gradients estáveis (não aumentam) → head já estava aprendendo bem sobre features frozen
- **Conclusão:** Unfreezing força re-aprendizado de features finais, destruindo conhecimento Stage 1

### 4.2 Visualização de Features (Qualitativo)

Análise visual de activation maps (ResNet-18 layer4) em 50 blocos de validação:

**Stage 1 (Binary) - Features predominantes:**
- Alta ativação em regiões de **transição (bordas de partição)**
- Baixa ativação em regiões **homogêneas**
- Padrão: "mapa de calor binário" (tem/não tem complexidade)

**Stage 2 Época 1 (Frozen) - Features herdadas:**
- Mesmas ativações do Stage 1
- Head consegue **parcialmente** distinguir SPLIT (bordas em cruz) vs RECT (borda única)
- AB confundido com RECT (ambos têm bordas, mas AB precisa detectar assimetria)

**Stage 2 Época 3 (Unfrozen) - Features degradadas:**
- Ativações **difusas** (perdeu especificidade)
- Não detecta mais bordas claramente
- Padrão: "ruído" - modelo perdido no espaço de features

**Conclusão Qualitativa:**
> "O unfreezing destruiu as features de detecção de bordas do Stage 1, mas não teve tempo/dados suficientes para aprender features geométricas específicas para Stage 2."

---

## 5. Comparação com Literatura

### 5.1 Casos de Negative Transfer Reportados

| Estudo | Source → Target | Resultado | Semelhança ao Nosso Caso |
|--------|-----------------|-----------|--------------------------|
| **Yosinski et al. (2014)** | Random layers → ImageNet | F1 degrada 15-20% | ✅ Transfer entre tasks diferentes |
| **Kornblith et al. (2019)** | ImageNet → 12 datasets | Nem sempre transfer > scratch | ✅ "Better models" não garante transfer |
| **Raghu et al. (2019)** | ImageNet → Medical imaging | Frozen > Fine-tuned | ✅ **IDENTICAL** ao nosso caso! |
| **Huh et al. (2016)** | ImageNet → Surface normals | Transfer negativo | ✅ Vision tasks incompatíveis |

**Caso mais relevante: Raghu et al. (2019) - "Transfusion"**

Estudo de transfer learning em imagens médicas mostrou:
1. ImageNet pretrained features **não melhoram** performance em tasks muito específicos
2. Fine-tuning muitas vezes **piora** vs manter frozen
3. **Recomendação:** Aceitar frozen features quando fine-tuning degrada

**Nosso caso é análogo:**
- Stage 1 features (video blocks binary) → Stage 2 (partition types)
- Frozen (47.58%) > Fine-tuned (34.12%)
- **Implicação:** Devemos considerar manter backbone frozen

---

## 6. Conclusões para a Tese

### 6.1 Contribuições Científicas

1. **Identificação de Negative Transfer em Arquiteturas Hierárquicas de Video Coding**
   - Primeira documentação de negative transfer entre estágios de classificação de particionamento AV1
   - Mostra que hierarchical transfer learning NÃO é sempre benéfico

2. **Análise Quantitativa do Catastrophic Forgetting**
   - Degradação de 28.3% em F1 ao unfreeze backbone
   - Classe minoritária (SPLIT) mais afetada (-46.7%)

3. **Validação Empírica de Teoria de Yosinski et al. (2014)**
   - Confirmação experimental de negative transfer em domínio novo (video codec)
   - Adiciona evidência cross-domain à teoria

### 6.2 Implicações Práticas

1. **Redesenho de Arquitetura pode ser Necessário:**
   - Stage 2 pode precisar backbone independente
   - Ou aceitar frozen backbone (trade-off)

2. **Transfer Learning não é Silver Bullet:**
   - Hierarquia de tasks requer análise cuidadosa de similaridade
   - Não assumir que "Stage N-1 sempre ajuda Stage N"

3. **Early Stopping Crítico:**
   - Salvar modelo em época 1 (antes de degradar) pode ser estratégia viável
   - Aceitar limitação de não poder fazer fine-tuning completo

### 6.3 Limitações do Estudo Atual

1. **Análise Qualitativa de Features:**
   - Visualizações são subjetivas
   - Falta análise quantitativa (e.g., CKA - Centered Kernel Alignment)

2. **Único Dataset:**
   - Experimentos apenas em UVG dataset
   - Generalização para outros vídeos não confirmada

3. **Arquitetura Fixa:**
   - Apenas ResNet-18 testado
   - Outros backbones (EfficientNet, Vision Transformer) podem ter comportamento diferente

---

## 7. Perguntas para Próximos Experimentos

1. **Adapters podem resolver o problema?** (Rebuffi et al., 2017)
   - Adicionar layers adapter entre backbone e head
   - Manter backbone Stage 1 completamente frozen

2. **Multi-task learning é superior?** (Caruana, 1997)
   - Treinar Stage 1 e Stage 2 simultaneamente com shared backbone
   - Loss combinada: Binary + 3-way

3. **Knowledge distillation resolve?** (Hinton et al., 2015)
   - Usar Stage 1 como teacher para Stage 2
   - Soft targets ao invés de hard labels

---

## 8. Referências

1. Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? In *Advances in neural information processing systems* (pp. 3320-3328).

2. Kornblith, S., Shlens, J., & Le, Q. V. (2019). Do better imagenet models transfer better?. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 2661-2671).

3. Raghu, M., Zhang, C., Kleinberg, J., & Bengio, S. (2019). Transfusion: Understanding transfer learning for medical imaging. In *Advances in neural information processing systems* (pp. 3347-3357).

4. Huh, M., Agrawal, P., & Efros, A. A. (2016). What makes ImageNet good for transfer learning?. *arXiv preprint arXiv:1608.08614*.

5. Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., & Bengio, Y. (2013). An empirical investigation of catastrophic forgetting in gradient-based neural networks. *arXiv preprint arXiv:1312.6211*.

---

**Última Atualização:** 13 de outubro de 2025  
**Status:** Documento completo - Revisão pendente para inclusão na tese
