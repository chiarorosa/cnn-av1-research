# Documentação V7 - Soluções para Tese

**Índice de Documentação Científica**

Este diretório contém a documentação técnico-científica de nível PhD para as três soluções arquiteturais propostas na pesquisa v7.

---

## 📚 Estrutura da Documentação

### **00_README.md** (este arquivo)
Índice e visão geral da documentação

### **01_solution1_conv_adapter.md**
- **Título:** "Parameter-Efficient Transfer Learning com Conv-Adapter para Predição de Partições AV1"
- **Literatura base:** Chen et al. (CVPR 2024)
- **Problema:** Negative transfer (Stage 2: 46%→32% F1 após unfreezing)
- **Solução:** Congelar backbone, aprender apenas modulação task-specific
- **Status:** 🚧 A ser documentado

### **02_solution2_ensemble.md**
- **Título:** "Ensemble Hierárquico Multi-Stage para Classificação Desbalanceada"
- **Literatura base:** Ahad et al. (2024)
- **Problema:** Erros individuais, classes raras (HORZ_4, VERT_4)
- **Solução:** Soft voting com 3 modelos diversos em CADA estágio
- **Status:** 🚧 A ser documentado

### **03_solution3_hybrid.md** ⭐
- **Título:** "Arquitetura Híbrida Adapter-Ensemble para Eficiência e Robustez"
- **Literatura base:** Combinação inovadora (Chen + Ahad)
- **Problema:** Trade-off entre eficiência e acurácia
- **Solução:** Ensemble de adapters (~10% params, F1 70-75%)
- **Status:** 🚧 A ser documentado

### **04_comparative_analysis.md**
- **Título:** "Análise Comparativa das Três Soluções"
- **Conteúdo:** 
  - Tabelas de métricas lado-a-lado
  - Análise estatística (significância)
  - Trade-offs (params vs accuracy vs speed)
  - Recomendações finais
- **Status:** 🚧 A ser documentado após experimentos

---

## 📋 Template para Cada Documento

Todos os documentos seguem a estrutura acadêmica padrão:

### **1. Resumo** (Abstract)
- 200-300 palavras
- Problema, solução, resultados principais

### **2. Introdução**
- Contextualização do problema
- Motivação (por que essa solução?)
- Objetivos específicos

### **3. Trabalhos Relacionados**
- 10-15 papers citados
- Estado-da-arte em cada sub-área
- Lacunas que motivam nossa abordagem

### **4. Fundamentação Teórica**
- Conceitos necessários
- Equações matemáticas
- Diagramas arquiteturais

### **5. Metodologia**
- Arquitetura proposta (detalhada)
- Implementação (pseudo-código, código real)
- Protocolo experimental (reprodutível)
- Hiperparâmetros

### **6. Experimentos**
- Setup experimental
- Datasets (split, augmentation)
- Métricas de avaliação
- Baseline comparativo

### **7. Resultados**
- Tabelas com métricas
- Gráficos (training curves, confusion matrices)
- Per-class performance
- Ablation studies

### **8. Análise e Discussão**
- Por que funcionou (ou falhou)?
- Comparação com literatura
- Limitações identificadas
- Trade-offs observados

### **9. Conclusões**
- Resumo dos achados
- Contribuições científicas
- Trabalhos futuros

### **10. Referências**
- Bibliografia completa (formato ABNT ou IEEE)

---

## 🎯 Critérios de Qualidade

Cada documento deve atender:

✅ **Rigor Científico:**
- Todas as afirmações fundamentadas em literatura ou experimentos
- Citações corretas (Autor, Ano)
- Reprodutibilidade (protocolos detalhados)

✅ **Clareza:**
- Figuras e tabelas auto-explicativas
- Nomenclatura consistente
- Código comentado

✅ **Contribuição:**
- Identificar claramente a inovação
- Comparar com estado-da-arte
- Justificar design choices

✅ **Completude:**
- Todos os experimentos documentados
- Resultados negativos incluídos (transparência)
- Limitações explicitadas

---

## 📊 Dados e Artefatos

Cada documento referencia:

1. **Checkpoints:**
   - `logs/v7_experiments/solution*/`
   - Modelos best e final
   - Training history

2. **Resultados:**
   - JSON com métricas
   - CSV com predições
   - Confusion matrices (PNG)

3. **Código:**
   - Scripts de treinamento (`scripts/0X0_*.py`)
   - Módulos (`v7_pipeline/*.py`)
   - Notebooks de análise (se houver)

4. **Figuras:**
   - Diagramas arquiteturais (draw.io, LaTeX TikZ)
   - Training curves (matplotlib)
   - Attention maps (para adapters)

---

## 📝 Workflow de Documentação

### **Durante Experimentos:**
1. Registrar hiperparâmetros em notebook/script
2. Salvar checkpoints com timestamp
3. Logar métricas por epoch
4. Capturar stderr/stdout (erros, warnings)

### **Após Experimentos:**
1. Gerar tabelas de resultados
2. Plotar gráficos comparativos
3. Escrever seção de Resultados
4. Fazer ablation analysis

### **Revisão Final:**
1. Verificar todas as citações
2. Revisar gramática/ortografia
3. Validar reprodutibilidade (executar script do zero)
4. Peer review interno (se possível)

---

## 🔗 Referências Externas

### **Papers Base:**
- Chen et al. (2024). "Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets". CVPR Workshop.
- Ahad et al. (2024). "A study on Deep Convolutional Neural Networks, Transfer Learning and Ensemble Model for Breast Cancer Detection".

### **Conceitos:**
- He et al. (2016). "Deep Residual Learning for Image Recognition". CVPR.
- Hu et al. (2018). "Squeeze-and-Excitation Networks". CVPR.
- Woo et al. (2018). "CBAM: Convolutional Block Attention Module". ECCV.

### **Contexto AV1:**
- Amestoy et al. (2022). "Tunable VVC Frame Partitioning Based on Lightweight Machine Learning". IEEE TIP.
- Li et al. (2021). "Fast CU Partition Decision for H.266/VVC Using CNN". IEEE Access.

---

## 🎓 Uso Acadêmico

Esta documentação será utilizada em:

1. **Tese de Doutorado** (capítulos 4-6)
2. **Papers para conferências/journals**
3. **Apresentações em bancas**
4. **Material suplementar para publicações**

**Licença:** Código e documentação para fins acadêmicos.  
**Contato:** chiarorosa@...

---

**Última atualização:** 14 de Outubro de 2025  
**Status:** 🚧 Estrutura criada, experimentos pendentes
