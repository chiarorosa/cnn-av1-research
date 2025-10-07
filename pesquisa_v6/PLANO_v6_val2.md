# PLANO V6 - Validação 2: Resolução do Stage 2

**Data:** 07 de outubro de 2025  
**Status:** Stage 2 apresentando Negative Transfer - necessária reimplementação

---

## 📋 Resumo Executivo

### Progresso Atual
- ✅ **Scripts 001-002:** Dataset preparado (152,600 train / 38,256 val)
- ✅ **Script 003 (Stage 1):** F1=72.28% (época 19) - **META ATINGIDA** ≥68%
- ⚠️ **Script 004 (Stage 2):** F1=46.51% (frozen) → 32-36% (unfrozen) - **PROBLEMA CRÍTICO**
- ✅ **Script 005 (Stage 3-RECT):** F1=68.44% (época 12)
- ✅ **Script 006 FGVC (Stage 3-AB):** F1=24.50% (4/4 classes, época 6)
- ✅ **Script 007 (Threshold):** threshold=0.45 → F1=72.79%
- ❌ **Script 008 (Pipeline):** F1=13.16% - **FALHA CASCATA** devido Stage 2
- ⏸️ **Script 009:** Bloqueado até resolução do Stage 2

### Meta Global
- Stage 1: F1 ≥ 68% ✅ **72.28%**
- Stage 2: F1 ≥ 45% ⚠️ **46.51% (frozen only)**
- Pipeline Final: Accuracy ≥ 48% ❌ **47.14% (pipeline quebrado)**

---

## 🔴 Problema Crítico: Stage 2 Catastrophic Forgetting

### Sintomas Observados

**Treinamento Original (antes ULMFiT):**
- Época 1 (frozen): F1=**47.58%** ✅
- Época 3 (unfrozen): F1=**34-38%** ❌
- Modelo salvo na época 1, causando falha do pipeline

**Treinamento com ULMFiT (atual):**
```
Época 1-8 (FROZEN):
  Best: F1=46.51% (época 1)
  - SPLIT: 40.75%
  - RECT: 60.66%
  - AB: 38.13%

🔓 Unfreezing backbone com Discriminative LR (época 9)
   Head LR: 5e-4
   Backbone LR: 1e-6 (500x menor)

Época 9-30 (UNFROZEN):
  F1=32-36% (oscilando)
  - SPLIT: 22% (degradou de 40%)
  - RECT: 37% (degradou de 60%)
  - AB: 42% (variável)
```

### Técnicas Tentadas (TODAS FALHARAM)

Implementamos **5 técnicas da literatura** sem sucesso:

1. **ULMFiT** (Howard & Ruder, 2018)
   - Gradual unfreezing: 8 épocas frozen → unfreezing
   - ❌ Não preveniu catastrophic forgetting

2. **Discriminative Fine-tuning** (Howard & Ruder, 2018)
   - Head LR: 5e-4
   - Backbone LR: 1e-6 (500x menor)
   - ❌ Mesmo com LR baixíssimo, backbone degrada

3. **Remoção de Label Smoothing** (Müller et al., 2019)
   - Paper: "When Does Label Smoothing Help?"
   - Conflito com Focal Loss removido
   - ❌ Não resolveu o problema

4. **Cosine Annealing** (Loshchilov & Hutter, 2017 - SGDR)
   - LR scheduling suave para melhor convergência
   - ❌ Convergência não é o problema

5. **ClassBalancedFocalLoss Device Fix** (Cui et al., 2019)
   - CB-beta=0.9999, gamma=2.0
   - Bug de device corrigido
   - ❌ Loss não era o problema

---

## 🔬 Diagnóstico Baseado em Literatura

### Negative Transfer (Yosinski et al., 2014)

**Paper:** "How transferable are features in deep neural networks?"

**Conceito-chave:**
- Transfer learning funciona quando **source task** e **target task** são similares
- Transfer learning **PREJUDICA** quando tasks são diferentes
- Features de layers finais são **task-specific**

### Análise Task Similarity: Stage 1 vs Stage 2

| Aspecto | Stage 1 | Stage 2 |
|---------|---------|---------|
| **Task** | Binary (NONE vs PARTITION) | 3-way (SPLIT vs RECT vs AB) |
| **Features necessárias** | "Tem partição?" | "Tipo de partição?" |
| **Foco visual** | Detecção de bordas de partição | Padrões de divisão geométrica |
| **Distribuição** | 50/50 balanceado | Long-tail (SPLIT minoritária) |
| **Complexidade** | Simples (presença/ausência) | Complexa (geometria) |

**Conclusão:** Tasks são **FUNDAMENTALMENTE DIFERENTES**

### Por que Backbone do Stage 1 Prejudica?

1. **Feature Specialization:**
   - Stage 1 backbone aprendeu a detectar "presença de particionamento"
   - Essas features suprimem padrões geométricos necessários para Stage 2
   
2. **Confirmation Bias:**
   - Kornblith et al., 2019: "Do Better ImageNet Models Transfer Better?"
   - Nem sempre transfer learning melhora performance
   - Às vezes, features genéricas (ImageNet) > features específicas (Stage 1)

3. **Catastrophic Forgetting:**
   - Goodfellow et al., 2013
   - Quando tentamos unfreeze, backbone tenta adaptar
   - Features do Stage 1 são destruídas, mas novas features não convergem

---

## 🎯 Soluções Propostas (Baseadas em Literatura)

### OPÇÃO 1: Train from Scratch ⭐ **RECOMENDADA**

**Referências:**
- Kornblith et al., 2019: "Do Better ImageNet Models Transfer Better?"
- He et al., 2019: "Rethinking ImageNet Pre-training"

**Implementação:**
```python
# ANTES (linha 283 do script 004):
checkpoint_stage1 = torch.load(stage1_model_path, ...)
model.backbone.load_state_dict(checkpoint_stage1['model_state_dict'])

# DEPOIS:
# Comentar linha acima - usar apenas ImageNet pretrained
# ResNet-18 com pretrained=True já foi inicializado na criação do modelo
```

**Parâmetros mantidos:**
- CB-Focal Loss (beta=0.9999, gamma=2.0)
- ULMFiT: 8 freeze epochs + discriminative LR
- Cosine Annealing scheduler
- 30 epochs total

**Expectativa:**
- F1 baseline (época 1 frozen): ~40-45%
- F1 após unfreezing: ~50-55% ✅ (sem catastrophic forgetting)
- Motivo: Backbone aprende features específicas para 3-way desde o início

**Vantagens:**
- ✅ Elimina conflito de features Stage 1 vs Stage 2
- ✅ ImageNet features são genéricas (edges, textures, shapes)
- ✅ Permite unfreezing sem degradação
- ✅ Implementação simples (comentar 1 linha)
- ✅ Base científica sólida

**Desvantagens:**
- ⚠️ Requer novo treinamento completo (30 epochs, ~2h)
- ⚠️ Pode ter baseline levemente menor (mas melhora depois)

---

### OPÇÃO 2: Frozen-Only Model

**Referências:**
- Raghu et al., 2019: "Transfusion: Understanding Transfer Learning"
- Mostra que frozen features às vezes são superiores

**Implementação:**
- Usar modelo da **época 1** (F1=46.51%)
- **NUNCA** fazer unfreezing do backbone
- Aceitar limitação

**Vantagens:**
- ✅ Modelo pronto AGORA
- ✅ F1=46.51% já atinge meta ≥45%
- ✅ Sem necessidade de retreinamento

**Desvantagens:**
- ❌ Limita potencial de melhoria
- ❌ Não resolve problema conceitual
- ❌ Backbone Stage 1 sempre será subótimo para Stage 2

---

### OPÇÃO 3: Hybrid Approach com Adapters

**Referências:**
- Rebuffi et al., 2017: "Learning multiple visual domains with residual adapters"
- Houlsby et al., 2019: "Parameter-Efficient Transfer Learning"

**Implementação:**
- Adicionar adapter layers entre backbone e head
- Treinar apenas adapters (backbone Stage 1 frozen)
- Permite adaptação sem modificar backbone

**Vantagens:**
- ✅ Mantém conhecimento do Stage 1
- ✅ Adapta para Stage 2 sem catastrophic forgetting

**Desvantagens:**
- ❌ Complexidade arquitetural alta
- ❌ Requer refatoração do modelo
- ❌ Validação científica incerta para este caso

---

### OPÇÃO 4: Redesenhar Hierarquia

**Referências:**
- Sun et al., 2017: "Revisiting Unreasonable Effectiveness of Data"

**Implementação:**
- Treinar Stage 2 direto nos dados ORIGINAIS (sem filtro Stage 1)
- Eliminar dependência hierárquica
- Usar Stage 1 apenas para roteamento no pipeline

**Vantagens:**
- ✅ Stage 2 vê todos os dados (não só PARTITION)
- ✅ Elimina viés do Stage 1

**Desvantagens:**
- ❌ Quebra conceito hierárquico do PLANO_V6.md
- ❌ Stage 2 precisa aprender a ignorar NONE (ruído)
- ❌ Aumenta complexidade do dataset

---

## 📊 Comparação das Opções

| Critério | Opção 1 (Scratch) | Opção 2 (Frozen) | Opção 3 (Adapters) | Opção 4 (Redesign) |
|----------|-------------------|------------------|--------------------|--------------------|
| **Implementação** | ⭐⭐⭐⭐⭐ Simples | ⭐⭐⭐⭐⭐ Imediata | ⭐⭐ Complexa | ⭐⭐⭐ Moderada |
| **Base Científica** | ⭐⭐⭐⭐⭐ Forte | ⭐⭐⭐⭐ Razoável | ⭐⭐⭐⭐ Válida | ⭐⭐⭐ Especulativa |
| **F1 Esperado** | ⭐⭐⭐⭐⭐ 50-55% | ⭐⭐⭐ 46.51% | ⭐⭐⭐⭐ 48-52% | ⭐⭐⭐⭐ 48-53% |
| **Tempo Implementação** | 5 min + 2h treino | 0 min | 2-3 dias | 1 dia |
| **Risco** | ⭐⭐⭐⭐⭐ Baixo | ⭐⭐⭐⭐ Baixo | ⭐⭐⭐ Médio | ⭐⭐ Alto |
| **Alinhamento PLANO_V6** | ⭐⭐⭐⭐⭐ Total | ⭐⭐⭐⭐ Bom | ⭐⭐⭐⭐ Bom | ⭐⭐ Requer revisão |

---

## 🚀 Recomendação Final

### **OPÇÃO 1: Train from Scratch** ⭐

**Razões:**
1. **Base científica sólida:** Kornblith et al. (2019) mostrou que ImageNet pretrained às vezes > task-specific transfer
2. **Simplicidade:** 1 linha de código comentada
3. **Melhor F1 esperado:** 50-55% (vs 46.51% atual)
4. **Resolve problema raiz:** Elimina negative transfer
5. **Permite unfreezing:** Sem catastrophic forgetting

**Plano de Ação:**

```bash
# 1. Backup do script atual
cp pesquisa_v6/scripts/004_train_stage2_redesigned.py \
   pesquisa_v6/scripts/004_train_stage2_redesigned_BACKUP.py

# 2. Modificar script 004 (comentar linha 283)
# ANTES:
#   checkpoint_stage1 = torch.load(...)
#   model.backbone.load_state_dict(checkpoint_stage1['model_state_dict'])
# DEPOIS:
#   # Usando apenas ImageNet pretrained (ResNet-18 padrão)
#   # Não carregar Stage 1 devido a negative transfer

# 3. Executar treinamento
source .venv/bin/activate
python pesquisa_v6/scripts/004_train_stage2_redesigned.py

# 4. Esperar ~2h (30 epochs)

# 5. Validar resultados:
#    - F1 frozen (épocas 1-8): esperar ~40-45%
#    - F1 unfrozen (épocas 9-30): esperar ~50-55%
#    - SEM degradação ao unfreeze

# 6. Se F1 ≥ 50%, prosseguir para script 008 (pipeline)
```

---

## 📝 Próximos Passos (Ordem de Execução)

### Fase 1: Resolver Stage 2 (CRÍTICO)
1. ✅ Decidir entre Opção 1, 2, 3 ou 4
2. ⏳ Implementar Opção 1 (5 minutos)
3. ⏳ Treinar Stage 2 from scratch (2 horas)
4. ⏳ Validar F1 ≥ 50%

### Fase 2: Re-executar Pipeline
5. ⏳ Script 008: Pipeline evaluation com novo Stage 2
   - Expectativa: F1 > 45%, Accuracy > 48%
6. ⏳ Analisar resultados pipeline completo
7. ⏳ Ajustar threshold se necessário (script 007)

### Fase 3: Comparação v5 vs v6
8. ⏳ Script 009: Comparação de desempenho
9. ⏳ Documentar resultados finais
10. ⏳ Atualizar README.md com pipeline completo

---

## 🔬 Referências Científicas Utilizadas

1. **Yosinski, J., et al. (2014).** "How transferable are features in deep neural networks?"  
   *NIPS 2014*  
   → Negative transfer entre tasks diferentes

2. **Kornblith, S., et al. (2019).** "Do Better ImageNet Models Transfer Better?"  
   *CVPR 2019*  
   → Nem sempre transfer learning melhora

3. **Howard, J., & Ruder, S. (2018).** "Universal Language Model Fine-tuning for Text Classification"  
   *ACL 2018*  
   → ULMFiT: gradual unfreezing + discriminative LR

4. **Goodfellow, I. J., et al. (2013).** "An Empirical Investigation of Catastrophic Forgetting"  
   *arXiv:1312.6211*  
   → Catastrophic forgetting em redes neurais

5. **Raghu, M., et al. (2019).** "Transfusion: Understanding Transfer Learning for Medical Imaging"  
   *NeurIPS 2019*  
   → Frozen features às vezes são superiores

6. **Müller, R., et al. (2019).** "When Does Label Smoothing Help?"  
   *NeurIPS 2019*  
   → Conflito label smoothing + Focal Loss

7. **Cui, Y., et al. (2019).** "Class-Balanced Loss Based on Effective Number of Samples"  
   *CVPR 2019*  
   → CB-Focal Loss para long-tail

8. **Loshchilov, I., & Hutter, F. (2017).** "SGDR: Stochastic Gradient Descent with Warm Restarts"  
   *ICLR 2017*  
   → Cosine annealing scheduler

9. **He, K., et al. (2019).** "Rethinking ImageNet Pre-training"  
   *ICCV 2019*  
   → Training from scratch pode igualar transfer learning

10. **Rebuffi, S. A., et al. (2017).** "Learning multiple visual domains with residual adapters"  
    *NeurIPS 2017*  
    → Adapter layers para multi-domain learning

---

## 📂 Arquivos Relevantes

### Scripts Treinamento
- `pesquisa_v6/scripts/003_train_stage1.py` - Stage 1 (F1=72.28% ✅)
- `pesquisa_v6/scripts/004_train_stage2_redesigned.py` - **MODIFICAR AQUI** ⚠️
- `pesquisa_v6/scripts/005_train_stage3_rect.py` - Stage 3-RECT (F1=68.44% ✅)
- `pesquisa_v6/scripts/006_train_stage3_ab_fgvc.py` - Stage 3-AB (F1=24.50% ✅)

### Scripts Avaliação
- `pesquisa_v6/scripts/007_optimize_threshold.py` - Threshold 0.45 ✅
- `pesquisa_v6/scripts/008_run_pipeline_eval_v6.py` - Pipeline (BLOQUEADO)
- `pesquisa_v6/scripts/009_compare_v5_v6.py` - Comparação (BLOQUEADO)

### Modelos Salvos
- `pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt` ✅
- `pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt` ⚠️ (época 1, frozen)
- `pesquisa_v6/logs/v6_experiments/stage3_rect/stage3_rect_model_best.pt` ✅
- `pesquisa_v6/logs/v6_experiments/stage3_ab/stage3_ab_fgvc_model_best.pt` ✅

### Históricos Treinamento
- `pesquisa_v6/logs/v6_experiments/stage2/stage2_history.pt` - 30 epochs completos
  - Época 1: F1=46.51% (frozen)
  - Épocas 9-30: F1=32-36% (unfrozen, degradado)

---

## 💡 Insights Aprendidos

### ✅ O que funcionou
1. **Stage 1:** Transfer learning de ImageNet → Binary task funcionou perfeitamente
2. **Stage 3:** Fine-tuning específico para cada tipo de partição (RECT, AB) eficaz
3. **FGVC Model:** Arquitetura com bilinear pooling melhorou classificação fine-grained
4. **CB-Focal Loss:** Lidou bem com long-tail distribution
5. **Threshold Optimization:** Ajuste fino melhorou F1 de 72.28% → 72.79%

### ❌ O que NÃO funcionou
1. **Stage 1 → Stage 2 Transfer:** Negative transfer entre binary e multi-class
2. **ULMFiT sozinho:** Não previne catastrophic forgetting quando tasks são diferentes
3. **Discriminative LR extremo:** Mesmo 500x de diferença não evitou degradação
4. **Label Smoothing:** Conflito com Focal Loss (paper Müller et al.)
5. **Unfreezing gradual:** Problema não é velocidade, mas compatibilidade de tasks

### 🔬 Lições para Pesquisa
1. **Task Similarity é CRÍTICO:** Avaliar similaridade antes de transfer learning
2. **Nem sempre transfer > scratch:** Kornblith et al. confirmado empiricamente
3. **Literatura é essencial:** Evitou tentativas aleatórias, focou em soluções validadas
4. **Monitoramento ativo:** Detectar degradação cedo evita desperdício de compute
5. **Pipelines hierárquicos:** Cada stage precisa ser independente ou muito similar

---

## 🎯 Critérios de Sucesso

### Stage 2 (após reimplementação)
- [ ] F1 ≥ 50% (target: 50-55%)
- [ ] SPLIT F1 ≥ 40%
- [ ] RECT F1 ≥ 55%
- [ ] AB F1 ≥ 45%
- [ ] **SEM degradação** ao unfreeze (F1 unfrozen ≥ F1 frozen)

### Pipeline Completo (script 008)
- [ ] Macro F1 ≥ 45%
- [ ] Accuracy ≥ 48%
- [ ] NONE: Precision/Recall ≥ 70%
- [ ] SPLIT: F1 ≥ 35%
- [ ] RECT: F1 ≥ 50%
- [ ] AB: F1 ≥ 40%

### Comparação v5 vs v6 (script 009)
- [ ] v6 F1 ≥ v5 F1
- [ ] v6 SPLIT F1 > v5 SPLIT F1 (foco principal)
- [ ] Documentação completa dos resultados

---

## 📅 Estimativa de Tempo

| Tarefa | Tempo Estimado |
|--------|----------------|
| Implementar Opção 1 | 5 minutos |
| Treinar Stage 2 (30 epochs) | 2 horas |
| Analisar resultados | 15 minutos |
| Re-executar Pipeline 008 | 30 minutos |
| Documentar resultados | 30 minutos |
| Script 009 comparação | 1 hora |
| **TOTAL** | **~4.5 horas** |

---

## 🔄 Histórico de Decisões

### 2025-10-07: Diagnóstico Negative Transfer
- **Problema:** Stage 2 degrada de 46.51% → 32-36% ao unfreeze
- **Tentativas:** 5 técnicas da literatura (ULMFiT, etc.) - todas falharam
- **Diagnóstico:** Negative transfer (Yosinski et al., 2014)
- **Decisão:** Recomendar train from scratch (Opção 1)

### 2025-10-07: Implementação ULMFiT
- **Mudanças:** 8 freeze epochs, discriminative LR (500x), remove label smoothing
- **Resultado:** F1 manteve em 46.51% (frozen), degradou igual
- **Conclusão:** Problema não é fine-tuning, mas inicialização

### 2025-10-06: Pipeline 008 Falha
- **Resultado:** F1=13.16%, Accuracy=47.14%
- **Causa:** Stage 2 salvo na época 1 (47.58%) do treinamento original
- **Impacto:** Cascata de erros nos Stages 3

---

## 📌 Notas Importantes

1. **Backup antes de modificar:** Sempre salvar versão atual antes de mudanças
2. **Validar história:** Checar `stage2_history.pt` após treinamento
3. **Monitorar unfreezing:** Epochs 9-10 são críticas para detectar degradação
4. **Threshold pode mudar:** Se Stage 2 melhorar, re-otimizar threshold
5. **Pipeline depende de TODOS:** Um stage ruim quebra toda hierarquia

---

## 📧 Contato

Para dúvidas ou discussão sobre as opções, revisar este documento antes de continuar.

**Última atualização:** 07 de outubro de 2025  
**Versão:** 2.0 - Diagnóstico Completo + Recomendações
