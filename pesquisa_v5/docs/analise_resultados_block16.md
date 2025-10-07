# Análise Detalhada dos Resultados - Blocos 16x16

## Resumo Executivo

Esta análise apresenta os resultados do pipeline hierárquico de predição de particionamento AV1 para blocos de tamanho 16x16, avaliado no conjunto de validação.

### Métrica Principal
- **Acurácia Final do Pipeline**: **39.56%**
- **Dataset**: `pesquisa/v5_dataset/block_16/val.pt`
- **Total de amostras**: ~65,603 blocos

---

## 1. Análise por Estágio

### Stage 1: Classificador Binário (Particiona vs Não-Particiona)

**Objetivo**: Identificar se um bloco deve ser particionado ou permanecer inteiro (PARTITION_NONE).

#### Métricas de Desempenho
| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| Acurácia | 62.69% | Acertos gerais moderados |
| Precisão | 53.71% | ~46% de falsos positivos ao prever particionamento |
| **Recall** | **82.93%** | **Excelente**: captura 83% dos blocos que devem ser particionados |
| F1-Score | 65.19% | Balanço razoável entre precisão e recall |

#### Matriz de Confusão (Stage 1)
- **True Positives (TP)**: 31,727 - blocos corretamente identificados como "devem particionar"
- **False Positives (FP)**: 27,347 - blocos NONE classificados erroneamente como "particionar"
- **False Negatives (FN)**: 6,529 - blocos particionados perdidos pelo classificador

#### Análise Crítica
✅ **Pontos Fortes**:
- Alto recall (82.93%) garante que a maioria dos blocos particionados são detectados
- Evita perda de informação estrutural importante

⚠️ **Pontos de Atenção**:
- Precisão baixa (53.71%) indica muitos falsos positivos
- ~27k blocos NONE são enviados erroneamente para Stage 2
- Isso sobrecarrega os estágios seguintes e propaga erros

**Impacto no Pipeline**:
- O alto recall é uma escolha de design conservadora: melhor processar demais do que perder partições complexas
- Threshold de 0.5 mantém esse comportamento (testado vs baseline)

---

### Stage 2: Classificação em Macro-Classes

**Objetivo**: Classificar blocos particionados em 5 macro-classes: NONE, SPLIT, RECT, AB, 1TO4.

#### Métricas Gerais
- **Macro F1-Score**: **33.41%** (baixo, indicando dificuldade na tarefa)

#### Matriz de Confusão (Stage 2)
```
Classes: [NONE, SPLIT, RECT, AB, 1TO4]

Predito →     NONE   SPLIT   RECT    AB    1TO4
Real ↓
NONE            0       0      0      0      0
SPLIT           0    3607    502   1853     0
RECT            0    1645  12065   4055     0
AB              0    3826   4079   6624     0
1TO4            0       0      0      0      0
```

#### Análise por Classe

**SPLIT (Partição Recursiva)**:
- Total: 5,962 amostras
- Acertos: 3,607 (60.5%)
- Confusão principal: 1,853 (31.1%) classificados como AB
- **F1 estimado**: ~50-55%

**RECT (Partições Horizontais/Verticais)**:
- Total: 17,765 amostras (classe majoritária)
- Acertos: 12,065 (67.9%)
- Confusões: 1,645 para SPLIT, 4,055 para AB
- **F1 estimado**: ~65-70%

**AB (Partições Assimétricas)**:
- Total: 14,529 amostras
- Acertos: 6,624 (45.6%)
- Confusões significativas: 3,826 para SPLIT, 4,079 para RECT
- **F1 estimado**: ~35-40% (pior desempenho)

**NONE e 1TO4**:
- Totalmente ausentes (0 predições e 0 amostras reais)
- NONE já foi filtrado em Stage 1
- 1TO4 aparentemente não existe no conjunto de validação

#### Problemas Identificados
1. **Confusão RECT ↔ AB**: 8,134 erros totais (4,055 + 4,079)
2. **Confusão SPLIT ↔ AB**: 5,679 erros totais (1,853 + 3,826)
3. **Classe AB com pior desempenho**: apenas 45.6% de acertos

---

### Stage 3: Especialistas

**Objetivo**: Refinar predições das macro-classes em partições específicas do AV1.

#### Stage 3 - RECT (Especialista para HORZ/VERT)

**Desempenho**: Macro F1 = **72.50%** ✅

Matriz de Confusão RECT:
```
Predito →     HORZ   VERT
Real ↓
HORZ         2127   1064
VERT          940   3360
```

**Análise**:
- Total: 7,491 amostras
- Acurácia: (2127+3360)/7491 = **73.2%**
- **HORZ**: Precisão 69.3% (2127/3067), Recall 66.7% (2127/3191)
- **VERT**: Precisão 75.9% (3360/4424), Recall 78.1% (3360/4300)
- Melhor desempenho em detectar partições verticais

#### Stage 3 - AB (Especialista para partições assimétricas)

**Desempenho**: Macro F1 = **25.26%** ⚠️ (CRÍTICO)

Matriz de Confusão AB:
```
Classes: [HORZ_A, HORZ_B, VERT_A, VERT_B]

Predito →   HORZ_A  HORZ_B  VERT_A  VERT_B
Real ↓
HORZ_A       799     23     154     644
HORZ_B       756     28     148     657
VERT_A       622     13     289     723
VERT_B       615     21     162     843
```

**Análise Crítica**:
- Total: 6,497 amostras
- **Problema grave**: O modelo colapsa para VERT_B
  - HORZ_A: apenas 49.4% acertos (799/1620)
  - HORZ_B: apenas 1.7% acertos (28/1589) ❌
  - VERT_A: apenas 17.5% acertos (289/1647) ❌
  - VERT_B: domina com 43.3% das predições (2867/6497)

**Diagnóstico**:
- **Viés extremo**: O modelo está enviando ~44% de todas as amostras para VERT_B
- Classes HORZ_B e VERT_A praticamente ignoradas
- Possível colapso de gradiente ou desbalanceamento severo não tratado
- Acurácia esperada: ~30-35% (muito baixo)

#### Stage 3 - 1TO4

Sem dados (matriz vazia) - classe não presente no dataset de validação.

---

## 2. Análise da Matriz de Confusão Final

### Distribuição das Classes (10 partições AV1)

```
Classes: 
0: PARTITION_NONE
1: PARTITION_HORZ
2: PARTITION_VERT
3: PARTITION_SPLIT
4: PARTITION_HORZ_A
5: PARTITION_HORZ_B
6: PARTITION_VERT_A
7: PARTITION_VERT_B
8: PARTITION_HORZ_4
9: PARTITION_VERT_4
```

### Distribuição Real das Classes (Total: 65,603)
| Classe | Total | Percentual |
|--------|-------|------------|
| PARTITION_NONE | 52,537 | 80.1% |
| PARTITION_HORZ | 8,147 | 12.4% |
| PARTITION_VERT | 9,618 | 14.7% |
| PARTITION_SPLIT | 5,962 | 9.1% |
| PARTITION_HORZ_A | 3,628 | 5.5% |
| PARTITION_HORZ_B | 3,537 | 5.4% |
| PARTITION_VERT_A | 3,794 | 5.8% |
| PARTITION_VERT_B | 3,570 | 5.4% |
| PARTITION_HORZ_4 | 0 | 0% |
| PARTITION_VERT_4 | 0 | 0% |

**Observação**: PARTITION_NONE domina com 80.1% dos dados.

### Acurácia por Classe

| Classe | Acertos | Total | Acurácia Individual |
|--------|---------|-------|---------------------|
| PARTITION_NONE | 25,190 | 52,537 | **47.9%** |
| PARTITION_HORZ | 2,127 | 8,147 | 26.1% |
| PARTITION_VERT | 3,360 | 9,618 | 34.9% |
| PARTITION_SPLIT | 3,286 | 5,962 | 55.1% |
| PARTITION_HORZ_A | 799 | 3,628 | 22.0% |
| PARTITION_HORZ_B | 28 | 3,537 | **0.8%** ❌ |
| PARTITION_VERT_A | 289 | 3,794 | **7.6%** ❌ |
| PARTITION_VERT_B | 843 | 3,570 | 23.6% |

### Principais Erros de Classificação

#### 1. PARTITION_NONE mal classificado (27,347 erros / 52.1% de erro)
- 9,934 → PARTITION_SPLIT (18.9%)
- 9,481 → PARTITION_VERT (18.0%)
- 4,931 → PARTITION_HORZ (9.4%)
- **Causa**: Falsos positivos do Stage 1 (baixa precisão)

#### 2. PARTITION_HORZ confusões (6,020 erros / 73.9% de erro)
- 2,544 → PARTITION_NONE (31.2%)
- 1,064 → PARTITION_VERT (13.1%)
- 1,014 → PARTITION_SPLIT (12.4%)
- **Causa**: Confusão em Stage 2 (RECT classificado como outras) + decisões incorretas em Stage 3

#### 3. PARTITION_VERT confusões (6,258 erros / 65.1% de erro)
- 2,272 → PARTITION_NONE (23.6%)
- 1,075 → PARTITION_VERT_B (11.2%)
- 940 → PARTITION_HORZ (9.8%)

#### 4. Classes AB com desempenho crítico
- **PARTITION_HORZ_B**: 3,509 erros (99.2% de erro!)
  - Distribuição caótica entre outras classes
- **PARTITION_VERT_A**: 3,505 erros (92.4% de erro!)
  - Maioria vai para PARTITION_SPLIT e PARTITION_VERT_B

---

## 3. Análise de Distribuição de Erros

### Propagação de Erros pelo Pipeline

1. **Stage 1 → Stage 2**:
   - 27,347 blocos NONE enviados incorretamente
   - Estes são distribuídos em Stage 2 entre SPLIT/RECT/AB
   - Poluem as estatísticas de Stage 3

2. **Stage 2 → Stage 3**:
   - 8,134 confusões RECT ↔ AB
   - Amostras AB enviadas para especialista RECT (e vice-versa)
   - Especialista AB recebe mix incorreto de dados

3. **Stage 3**:
   - Especialista RECT: razoável (72.5% F1)
   - Especialista AB: colapso total (25.3% F1)

### Padrões Identificados

**Sobre-predição**:
- PARTITION_SPLIT: recebe muitos falsos positivos de NONE
- PARTITION_VERT_B: viés extremo no especialista AB

**Sub-predição**:
- PARTITION_HORZ_B: praticamente não é predito
- PARTITION_VERT_A: severamente sub-representado

---

## 4. Diagnóstico e Recomendações

### Problemas Críticos Identificados

1. **Stage 1: Precisão Baixa (53.71%)**
   - **Impacto**: 27k falsos positivos contaminam todo o pipeline
   - **Solução proposta**:
     - Aumentar threshold para 0.6-0.7 (trade-off: perder alguns verdadeiros positivos)
     - Implementar calibração de probabilidades
     - Adicionar features discriminativas para NONE

2. **Stage 2: Confusão entre Macro-Classes**
   - **Impacto**: Apenas 33.41% Macro F1
   - **Solução proposta**:
     - Aumentar separabilidade entre RECT e AB
     - Features específicas: análise de gradientes direcionais
     - Data augmentation diferenciado por classe
     - Penalizar mais confusões críticas (RECT ↔ AB)

3. **Stage 3 AB: Colapso do Modelo (25.26% F1)**
   - **Impacto**: Classes HORZ_B e VERT_A perdidas
   - **Causas prováveis**:
     - Desbalanceamento severo não tratado
     - Falta de features discriminativas
     - Possível overfitting para VERT_B
   - **Soluções propostas**:
     - **Imediatas**:
       - Implementar weighted sampling mais agressivo
       - Focal loss com gamma alto (2.0-3.0)
       - Class weights baseados em inverse frequency
     - **Médio prazo**:
       - Separar em 2 especialistas: um para HORZ_A/B, outro para VERT_A/B
       - Feature engineering específico (ângulos de gradiente, análise espectral)
       - Ensemble de modelos com diferentes inicializações
     - **Arquitetura**:
       - Testar arquiteturas attention-based
       - Multi-task learning com tarefas auxiliares

4. **Distribuição de Dados Desbalanceada**
   - 80% PARTITION_NONE domina o dataset
   - **Soluções**:
     - Undersampling de NONE no treino
     - Oversampling sintético (SMOTE, mixup) para classes raras
     - Stratified sampling por classe

### Métricas Alvo Realistas

Baseado na análise, métricas alcançáveis com melhorias:

| Estágio | Métrica Atual | Meta Curto Prazo | Meta Longo Prazo |
|---------|---------------|------------------|------------------|
| Stage 1 F1 | 65.19% | 68-70% | 72-75% |
| Stage 2 Macro F1 | 33.41% | 40-45% | 50-55% |
| Stage 3 RECT F1 | 72.50% | 75-78% | 80-82% |
| Stage 3 AB F1 | 25.26% | 45-50% ⚠️ | 60-65% |
| **Acurácia Final** | **39.56%** | **48-52%** | **58-62%** |

### Priorização de Ações

**Prioridade 1 (Crítico - resolver primeiro)**:
1. Consertar Stage 3 AB (maior gargalo)
   - Re-treinar com focal loss gamma=2.5
   - Implementar balanced sampling
   - Verificar distribuição do dataset Stage 3 AB

**Prioridade 2 (Alto impacto)**:
2. Melhorar Stage 1 precisão
   - Calibrar threshold otimizado
   - Adicionar regularização
3. Reduzir confusão RECT ↔ AB em Stage 2
   - Features direcionais
   - Penalização customizada

**Prioridade 3 (Refinamento)**:
4. Balanceamento de dataset global
5. Feature engineering avançado
6. Ensemble de modelos

---

## 5. Análise Qualitativa

### Pontos Fortes do Pipeline Atual

✅ **Arquitetura Hierárquica Funcional**:
- Pipeline de 3 estágios implementado e operacional
- Especialistas RECT e AB em funcionamento

✅ **Stage 1 com Alto Recall**:
- 82.93% de recall garante captura de partições importantes
- Estratégia conservadora adequada para não perder informação

✅ **Especialista RECT com Bom Desempenho**:
- 72.5% F1 é aceitável para tarefa de classificação binária
- Diferenciação HORZ vs VERT relativamente robusta

### Limitações e Desafios

⚠️ **Propagação de Erros**:
- Cada estágio amplifica erros do anterior
- 27k falsos positivos em Stage 1 poluem todo o pipeline

⚠️ **Desbalanceamento Severo**:
- 80% PARTITION_NONE domina
- Classes AB sub-representadas no treino final

⚠️ **Especialista AB Crítico**:
- Colapso em 2 das 4 classes (HORZ_B, VERT_A)
- Viés extremo para VERT_B
- Requer redesign completo

---

## 6. Conclusão

O pipeline hierárquico de predição de particionamento AV1 para blocos 16x16 apresenta **resultados preliminares funcionais, mas com necessidade crítica de melhorias**, especialmente no Stage 3 AB.

### Resumo dos Resultados
- ✅ **Acurácia Final**: 39.56% (baseline funcional)
- ✅ **Stage 1**: 65.19% F1 (recall alto, precisão a melhorar)
- ⚠️ **Stage 2**: 33.41% Macro F1 (confusões significativas)
- ✅ **Stage 3 RECT**: 72.50% F1 (bom desempenho)
- ❌ **Stage 3 AB**: 25.26% F1 (CRÍTICO - requer atenção imediata)

### Próximos Passos Recomendados
1. **Imediato**: Re-treinar Stage 3 AB com focal loss e balanced sampling
2. **Curto prazo**: Otimizar threshold Stage 1 e melhorar separação RECT/AB em Stage 2
3. **Médio prazo**: Feature engineering e arquitetura mais robusta para classes AB
4. **Longo prazo**: Considerar ensemble ou arquiteturas attention-based

A arquitetura hierárquica é promissora, mas requer ajustes significativos em balanceamento de classes e capacidade discriminativa dos especialistas, particularmente para partições assimétricas.

---

**Análise gerada em**: 06/10/2025
**Dataset**: `pesquisa/v5_dataset/block_16/val.pt`
**Configuração**: Threshold Stage1=0.5, Especialistas RECT+AB
