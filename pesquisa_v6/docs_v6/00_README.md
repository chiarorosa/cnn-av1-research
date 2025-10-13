# Documentação Técnico-Científica - Pesquisa v6

**Objetivo:** Documentar experimentos, metodologias, resultados e análises para futura Tese de Doutorado

**Data de Criação:** 13 de outubro de 2025  
**Pesquisador:** Chiaro Rosa  
**Projeto:** CNN-AV1 - Predição de Particionamento Hierárquico para Codec AV1

---

## 📚 Estrutura da Documentação

### 1. Fundamentos
- `01_problema_negative_transfer.md` - Análise do problema de negative transfer no Stage 2
- `02_literatura_base.md` - Revisão bibliográfica das técnicas aplicadas

### 2. Experimentos
- `03_experimento_ulmfit.md` - Tentativa de solução com ULMFiT (Howard & Ruder, 2018)
- `04_experimento_train_from_scratch.md` - Train from Scratch (Kornblith et al., 2019)
- `05_avaliacao_pipeline_completo.md` - ✅ **Avaliação End-to-End do Pipeline Hierárquico**

### 3. Análises
- `06_analise_comparativa.md` - Comparação quantitativa dos modelos
- `07_insights_e_conclusoes.md` - Insights científicos e conclusões

### 4. Metodologia
- `07_metodologia_experimental.md` - Protocolos, datasets, métricas

---

## 🎯 Resumo Executivo

### Problema Central
Stage 2 (classificador 3-way: SPLIT, RECT, AB) apresenta **catastrophic forgetting** quando inicializado com backbone do Stage 1 (classificador binário: NONE vs PARTITION).

### Hipótese Inicial
Negative transfer entre tarefas dissimilares (Yosinski et al., 2014):
- Stage 1: Features para detecção de particionamento (binário)
- Stage 2: Features para classificação de tipos de partição (3-way)

### Experimentos Realizados

#### Experimento 1: ULMFiT (FALHOU)
- **Técnicas:** Gradual unfreezing, discriminative LR, cosine annealing, CB-Focal Loss
- **Resultado:** F1=46.51% (frozen) → 32-36% (unfrozen) ❌
- **Conclusão:** Técnicas de fine-tuning não resolvem incompatibilidade de features

#### Experimento 2: Train from Scratch (PARCIALMENTE SUCEDIDO)
- **Técnica:** Treinar Stage 2 com ImageNet pretrained (sem Stage 1)
- **Resultado:** F1=8.99% (frozen) → 37.38% (unfrozen) ✅ sem degradação
- **Conclusão:** Elimina catastrophic forgetting, mas F1 inferior ao Stage 1 init (46.51%)

### Insights Científicos

1. **Stage 1 features SÃO úteis** (46.51% > 37.38% ImageNet-only)
2. **ULMFiT insuficiente** para adaptar features entre tarefas hierárquicas
3. **Frozen-only viável** - F1=46.51% atinge meta (≥45%)
4. **Trade-off confirmado:** Adaptabilidade vs Performance inicial

### Próximas Direções

1. **Opção 2 (Recomendada):** Usar Stage 1 model frozen (F1=46.51%)
2. **Opção 3:** Adapter layers (Rebuffi et al., 2017)
3. **Opção 4:** Arquitetura multi-task com shared backbone

---

## 📖 Guia de Leitura

### Para Tese - Capítulo de Metodologia
1. Ler `07_metodologia_experimental.md`
2. Ler `02_literatura_base.md`

### Para Tese - Capítulo de Resultados
1. Ler `03_experimento_ulmfit.md`
2. Ler `04_experimento_train_from_scratch.md`
3. Ler `05_analise_comparativa.md`

### Para Tese - Capítulo de Discussão
1. Ler `01_problema_negative_transfer.md`
2. Ler `06_insights_e_conclusoes.md`

---

## 🔬 Dados Experimentais

### Datasets
- **Train:** 152,600 amostras (SPLIT: 23,942 | RECT: 71,378 | AB: 57,280)
- **Val:** 38,256 amostras
- **Formato:** Blocos 16×16 pixels YUV 4:2:0 10-bit (codificados como float32)

### Hardware
- **GPU:** NVIDIA RTX (CUDA)
- **Tempo por época:** ~15s (frozen), ~20s (unfrozen)

### Repositório
- **Branch:** `feat/stage2-train-from-scratch`
- **Scripts:** `pesquisa_v6/scripts/004_train_stage2_redesigned.py`
- **Logs:** `pesquisa_v6/logs/v6_experiments/stage2_scratch/`

---

## 📅 Timeline

- **07/10/2025:** Identificação do problema (catastrophic forgetting)
- **07/10/2025:** Experimento 1 - ULMFiT (falhou)
- **13/10/2025:** Experimento 2 - Train from Scratch (parcial)
- **13/10/2025:** Documentação técnico-científica criada
- **13/10/2025:** ✅ **Pipeline completo avaliado - Decisão final tomada**

---

## 📊 Figuras e Tabelas

As figuras estão localizadas em cada documento específico. Para inclusão na tese:

- **Tabela 1:** Comparação Stage 1 init vs ImageNet-only (`06_analise_comparativa.md`)
- **Tabela 2:** ⭐ Comparação Pipeline Frozen vs Scratch (`05_avaliacao_pipeline_completo.md`)
- **Figura 1:** Evolução F1 por época (`04_experimento_train_from_scratch.md`)
- **Figura 2:** Arquitetura hierárquica (`01_problema_negative_transfer.md`)
- **Figura 3:** ⭐ Erro em cascata RECT/AB (`05_avaliacao_pipeline_completo.md`)

---

## 🔗 Referências Principais

1. Yosinski et al. (2014) - "How transferable are features in deep neural networks?"
2. Howard & Ruder (2018) - "Universal Language Model Fine-tuning" (ULMFiT)
3. Kornblith et al. (2019) - ⭐ **VALIDADO** - "Do Better ImageNet Models Transfer Better?"
4. Cui et al. (2019) - "Class-Balanced Loss Based on Effective Number of Samples"
5. Raghu et al. (2019) - "Transfusion: Understanding Transfer Learning"
6. He et al. (2016) - ⭐ **VALIDADO** - "Deep Residual Learning for Image Recognition"
7. Sun et al. (2017) - ⭐ **VALIDADO** - "Revisiting Unreasonable Effectiveness of Data"

---

**Última Atualização:** 13 de outubro de 2025  
**Status:** ✅ **CONCLUÍDO** - Pipeline V6 otimizado com Train from Scratch (Accuracy 47.66%, meta 48%)
