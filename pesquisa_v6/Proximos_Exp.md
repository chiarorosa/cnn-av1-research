# Próximos Experimentos - Pipeline V6

**Data:** 13 de outubro de 2025  
**Status:** Planejamento Estratégico  
**Responsável:** @chiarorosa  
**Nível:** PhD - Fundamentado em Literatura Científica

---

## 📋 Sumário Executivo

Este documento apresenta os experimentos com **maior potencial de melhoria** para o Pipeline V6, baseado na análise completa da documentação (`docs_v6/`) e descobertas até a presente data.

**Situação Atual:**
- Pipeline Accuracy: **47.66%** (meta: 48.0%, gap: -0.34pp)
- Stage 2 colapsado descoberto (prediz 100% RECT ou 99.99% SPLIT)
- Erro cascata Stage 3: -93% degradação (standalone 68%/24% → pipeline 4%/1.5%)
- Noise Injection (Exp 09): Sucesso parcial (-1.80pp accuracy, mas +28pp/+56pp cascade error)

---

## 🔴 **PRIORIDADE CRÍTICA** - Resolver Stage 2 Colapsado

### **Exp 10A: Recuperar/Validar Modelo Stage 2 Frozen (Época 0)**

**Problema Identificado:**
- Checkpoint `stage2_model_best.pt` está colapsado (prediz RECT em 100% das amostras)
- Checkpoint `stage2_model_final.pt` também colapsado (prediz SPLIT em 99.99%)
- Análise do history mostra época 0 (frozen) tinha F1=46.51% ✅
- Época 8 (após unfreeze) colapsou para F1=34.39% ❌

**Hipótese:**
> "O modelo frozen (época 0) funciona corretamente (F1=46.51%). Catastrophic forgetting ao unfreeze destruiu features. Solução: usar modelo frozen."

**Protocolo:**

1. **Verificar existência de checkpoint frozen:**
   ```bash
   ls -lh pesquisa_v6/logs/v6_experiments/stage2/ | grep -E "ep[0-9]|frozen"
   ```

2. **Se não existir, retreinar 1 época frozen:**
   ```bash
   python3 pesquisa_v6/scripts/004_train_stage2_redesigned.py \
     --dataset-dir pesquisa_v6/v6_dataset/block_16 \
     --epochs 1 \
     --batch-size 128 \
     --output-dir pesquisa_v6/logs/v6_experiments/stage2_frozen \
     --device cuda \
     --save-every-epoch
   ```

3. **Validar modelo frozen com Script 009:**
   ```bash
   python3 pesquisa_v6/scripts/009_analyze_stage2_confusion.py \
     --stage2-model <path_to_frozen_model> \
     --dataset-dir pesquisa_v6/v6_dataset/block_16 \
     --device cuda
   ```
   **Esperado:** F1 ~0.46-0.47, accuracy ~48-49%

4. **Re-avaliar Pipeline Experimento 09:**
   ```bash
   python3 pesquisa_v6/scripts/008_run_pipeline_eval_v6.py \
     --stage1-model pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt \
     --stage2-model <frozen_model> \
     --stage3-rect-model <robust_rect_model> \
     --stage3-ab-models <ab_ensemble> \
     --output-dir pesquisa_v6/logs/v6_experiments/pipeline_eval_frozen_s2
   ```

**Potencial de Ganho:** +0 a +2pp (desbloqueia demais experimentos)  
**Esforço:** Baixo (1 hora)  
**Risco:** Baixo  
**Fundamentação:** Kornblith et al. (2019) - Features congeladas podem superar fine-tuning em tasks dissimilares  
**Status:** **🚨 BLOQUEADOR** - Todos experimentos Stage 3 dependem disso

**Documentação:** Atualizar `docs_v6/10_stage2_collapse_resolution.md`

---

## 🟠 **ALTA PRIORIDADE** - Robustez do Pipeline

### **Exp 10B: Confusion-Based Noise Injection**

**Objetivo:** Substituir labels aleatórios por confusão real do Stage 2 para treinamento mais realista do Stage 3.

**Motivação:**
- Exp 09 usou noise aleatório (25% labels aleatórios)
- Stage 2 tem padrões de confusão específicos (ex: RECT→AB em X%, AB→SPLIT em Y%)
- Treinar Stage 3 com distribuição real de erros deve melhorar robustez

**Técnica:**

1. **Extrair confusion matrix do Stage 2 frozen:**
   - Usar Script 009 para gerar matriz normalizada
   - Exemplo: `P(prediz AB | GT=RECT) = 0.30`

2. **Implementar `ConfusionBasedNoisyDataset`:**
   ```python
   class ConfusionBasedNoisyDataset(Dataset):
       def __init__(self, clean_dataset, confusion_matrix, noise_ratio=0.25):
           self.clean_dataset = clean_dataset
           self.confusion_matrix = confusion_matrix  # 3x3 (SPLIT, RECT, AB)
           self.noise_ratio = noise_ratio
       
       def __getitem__(self, idx):
           block, gt_label, qp = self.clean_dataset[idx]
           
           if random.random() < self.noise_ratio:
               # Sample label confuso baseado em confusion matrix
               noisy_label = np.random.choice(
                   [0, 1, 2], 
                   p=self.confusion_matrix[gt_label]
               )
               # Mapear para Stage 3 labels
               mapped_label = self.map_stage2_to_stage3(noisy_label)
               return block, mapped_label, qp
           else:
               return block, gt_label, qp  # Clean
   ```

3. **Retreinar Stage 3-RECT e Stage 3-AB:**
   - Modificar Scripts 005 e 006 para usar `ConfusionBasedNoisyDataset`
   - Manter hiperparâmetros do Exp 09

4. **Avaliar pipeline completo:**
   - Comparar com Exp 09 baseline (45.86%)

**Potencial de Ganho:** +1.5 a +3pp (45.86% → 47.5-49%)  
**Esforço:** Médio (1-2 dias)  
**Risco:** Baixo  
**Fundamentação:** 
- Natarajan et al. (2013) - Learning with Noisy Labels: distribuição realista > aleatória
- Heigold et al. (2016) - Cascade systems: treinar com erros upstream melhora robustez

**Requisito:** Exp 10A concluído (Stage 2 funcional)

**Documentação:** `docs_v6/10_confusion_based_noise_injection.md`

---

### **Exp 10C: Train-with-Predictions (Real Distribution)**

**Objetivo:** Stage 3 treina com predições reais do Stage 2 em tempo real (não noise sintético).

**Diferença vs Exp 10B:**
- 10B: Usa confusion matrix (estatística agregada)
- 10C: Usa predição real do Stage 2 para cada sample

**Técnica:**

```python
class TrainWithPredictionsDataset(Dataset):
    def __init__(self, stage2_model, stage3_dataset, clean_ratio=0.75):
        self.stage2_model = stage2_model.eval()
        self.stage3_dataset = stage3_dataset
        self.clean_ratio = clean_ratio
    
    def __getitem__(self, idx):
        block, gt_label, qp = self.stage3_dataset[idx]
        
        if random.random() < self.clean_ratio:
            return block, gt_label, qp  # 75% clean (GT)
        else:
            # Computar predição Stage 2 (frozen, sem grad)
            with torch.no_grad():
                stage2_logits = self.stage2_model(block.unsqueeze(0))
                stage2_pred = torch.argmax(stage2_logits, dim=1).item()
            
            # Mapear predição Stage 2 → Stage 3 labels
            # SPLIT (0) → skip (Stage 3 não processa)
            # RECT (1) → treinar Stage 3-RECT
            # AB (2) → treinar Stage 3-AB
            mapped_label = self.map_stage2_to_stage3(stage2_pred, head='RECT')
            return block, mapped_label, qp  # 25% noisy (Stage 2 pred)
```

**Vantagens:**
- ✅ Distribuição **exata** de inferência (não aproximação)
- ✅ Adapta automaticamente se Stage 2 melhorar
- ✅ Não precisa analisar confusion matrix

**Desvantagens:**
- ❌ Mais lento (forward pass Stage 2 por sample)
- ❌ Requer Stage 2 carregado em memória

**Potencial de Ganho:** +2 a +4pp (melhor que Exp 10B)  
**Esforço:** Médio (2-3 dias)  
**Risco:** Médio (lentidão pode ser proibitiva)  
**Fundamentação:** Heigold et al. (2016) - "Training with predictions of upstream models is optimal for cascade systems"

**Alternativa a:** Exp 10B (escolher um dos dois)

**Documentação:** `docs_v6/10_train_with_predictions.md`

---

### **Exp 10D: Ensemble Real para Stage 3-AB**

**Problema Identificado:**
- Exp 09 usou 3 cópias do **mesmo modelo** AB (não é ensemble verdadeiro)
- Ensemble requer modelos **independentes** (diferentes inicializações/seeds)

**Técnica:**

1. **Treinar 3 modelos AB independentes:**
   ```bash
   for seed in 42 123 456; do
     python3 pesquisa_v6/scripts/006_train_stage3_ab_fgvc.py \
       --seed $seed \
       --output-dir pesquisa_v6/logs/v6_experiments/stage3_ab_robust_seed${seed} \
       --noise-injection 0.25 \
       --noise-sources RECT SPLIT
   done
   ```

2. **Implementar ABEnsemble com majority voting:**
   ```python
   class ABEnsemble(nn.Module):
       def __init__(self, models):
           super().__init__()
           self.models = nn.ModuleList(models)
       
       def forward(self, x):
           preds = [model(x) for model in self.models]
           # Soft voting (average probabilities)
           avg_probs = torch.stack(preds).mean(dim=0)
           return avg_probs
   ```

3. **Avaliar pipeline com ensemble real:**
   - Comparar com Exp 09 (1 modelo repetido 3x)

**Potencial de Ganho:** +1 a +2pp overall, +5 a +8pp F1 AB  
**Esforço:** Baixo (1 dia, 3× treinamento paralelo)  
**Risco:** Baixo  
**Fundamentação:** Dietterich (2000) - "Ensemble Methods in Machine Learning": ensemble de modelos independentes reduz variância

**Requisito:** Exp 10A concluído

**Documentação:** Atualizar `docs_v6/09_noise_injection_stage3.md` seção "Limitações"

---

## 🟡 **MÉDIA PRIORIDADE** - Melhorar Stage 2

### **Exp 11A: Adapter Layers no Stage 2**

**Objetivo:** Permitir fine-tuning do backbone Stage 1 sem catastrophic forgetting.

**Problema Atual:**
- Backbone Stage 1 tem features úteis (F1 frozen=46.51%)
- Unfreezing causa catastrophic forgetting (F1=34%)
- Técnicas ULMFiT falharam (doc `03_experimento_ulmfit.md`)

**Solução Proposta: Adapter Layers (Rebuffi et al., 2017)**

**Arquitetura:**
```
ResNet-18 Block (frozen Stage 1 weights)
    ↓
[Conv → BN → ReLU]  ← Frozen
    ↓
+ Adapter Layer (trainable)  ← NEW!
    ↓
[Conv → BN → ReLU]  ← Frozen
    ↓
+ Adapter Layer (trainable)  ← NEW!
```

**Adapter Layer:**
```python
class AdapterLayer(nn.Module):
    def __init__(self, in_features, bottleneck_dim=64):
        super().__init__()
        self.down = nn.Linear(in_features, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, in_features)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Residual connection
        adapter_output = self.up(self.relu(self.down(x)))
        return x + adapter_output  # Skip connection
```

**Implementação:**

1. **Modificar `Stage2Model` para incluir adapters:**
   - Inserir `AdapterLayer` após cada `layer1, layer2, layer3, layer4`
   - Total: 4 adapters × 2 layers (down+up) = 8 layers adicionais
   - Parâmetros: ~50k (vs 11M do backbone)

2. **Treinar apenas adapters + head:**
   ```python
   # Freeze backbone Stage 1
   for param in model.backbone.parameters():
       param.requires_grad = False
   
   # Unfreeze adapters
   for name, param in model.named_parameters():
       if 'adapter' in name or 'head' in name:
           param.requires_grad = True
   
   optimizer = AdamW([
       {'params': [p for n, p in model.named_parameters() 
                   if 'adapter' in n], 'lr': 1e-4},
       {'params': model.head.parameters(), 'lr': 5e-4}
   ])
   ```

3. **Protocolo de treinamento:**
   - Epochs: 50
   - Batch size: 128
   - CB-Focal Loss (gamma=2.0, beta=0.9999)
   - Cosine annealing scheduler

**Potencial de Ganho:** +5 a +10pp Stage 2 F1 (46% → 51-56%), +2 a +5pp pipeline accuracy  
**Esforço:** Alto (1 semana implementação + treinamento)  
**Risco:** Médio (pode não convergir, adapters podem ser insuficientes)  
**Fundamentação:** 
- Rebuffi et al. (2017) - "Learning multiple visual domains with residual adapters": adapters resolvem negative transfer
- Houlsby et al. (2019) - "Parameter-Efficient Transfer Learning for NLP": adapters < 1% parâmetros, 95%+ performance

**Se funcionar:** 🎯 **GAME CHANGER** - resolve problema central do projeto

**Documentação:** `docs_v6/11_adapter_layers_stage2.md`

---

### **Exp 11B: Meta-Learning (MAML/Reptile) para Stage 2**

**Objetivo:** Aprender inicialização ótima do backbone Stage 1 que facilite adaptação para Stage 2.

**Problema Atual:**
- Backbone Stage 1 otimizado para task binária (NONE vs PARTITION)
- Fine-tuning para task 3-way (SPLIT vs RECT vs AB) causa forgetting

**Solução: Model-Agnostic Meta-Learning (MAML, Finn et al., 2017)**

**Ideia:**
> "Treinar Stage 1 para que suas features sejam **rapidamente adaptáveis** para Stage 2 (não apenas boas para binary)."

**Algoritmo:**

1. **Meta-training Stage 1:**
   ```python
   # Para cada batch:
   # 1. Clone model
   model_clone = deepcopy(model_stage1)
   
   # 2. Simular adaptação para Stage 2 (few gradient steps)
   for _ in range(K=5):  # Inner loop
       loss_stage2 = compute_stage2_loss(model_clone)
       model_clone.update(loss_stage2)
   
   # 3. Computar meta-loss (quão bem adaptou?)
   meta_loss = evaluate_stage2_performance(model_clone)
   
   # 4. Atualizar model_stage1 original (outer loop)
   model_stage1.backward(meta_loss)
   ```

2. **Após meta-training:**
   - Stage 1 aprendeu features que são **boas para binary** E **fáceis de adaptar para 3-way**

3. **Fine-tuning Stage 2:**
   - Usar Stage 1 meta-trained como init
   - Adaptar para Stage 2 (deve evitar catastrophic forgetting)

**Potencial de Ganho:** +3 a +8pp Stage 2 F1  
**Esforço:** Muito Alto (2 semanas - complexo)  
**Risco:** Alto (MAML notoriamente difícil de implementar, pode não convergir)  
**Fundamentação:** 
- Finn et al. (2017) - "Model-Agnostic Meta-Learning": MAML resolve task mismatch
- Nichol et al. (2018) - "Reptile": versão simplificada do MAML, mais estável

**Alternativa mais simples:** Reptile (Nichol et al., 2018) - não requer second-order derivatives

**Documentação:** `docs_v6/11_meta_learning_stage2.md`

---

### **Exp 11C: Knowledge Distillation Stage 1 → Stage 2**

**Objetivo:** Transferir conhecimento de Stage 1 sem copiar pesos (evita negative transfer).

**Problema Atual:**
- Copiar pesos Stage 1 → Stage 2 causa negative transfer
- Train from scratch (ImageNet-only) tem F1 inferior (37.38% vs 46.51%)

**Solução: Knowledge Distillation (Hinton et al., 2015)**

**Técnica:**

```python
# Stage 2 aprende a imitar features intermediárias do Stage 1
def distillation_loss(stage1_model, stage2_model, x, y_stage2):
    # 1. Forward Stage 1 (frozen, apenas para extrair features)
    with torch.no_grad():
        features_stage1 = stage1_model.backbone(x)  # (B, 512)
    
    # 2. Forward Stage 2 (trainable)
    features_stage2 = stage2_model.backbone(x)  # (B, 512)
    logits_stage2 = stage2_model.head(features_stage2)
    
    # 3. Loss combinada
    loss_task = CE(logits_stage2, y_stage2)  # Task loss (3-way)
    loss_distill = MSE(features_stage2, features_stage1)  # Feature mimicking
    
    return loss_task + λ * loss_distill  # λ=0.5
```

**Vantagens:**
- ✅ Stage 2 aprende de Stage 1 sem copiar pesos (não há catastrophic forgetting)
- ✅ Features Stage 1 atuam como "regularizador suave" (não hard constraint)
- ✅ Stage 2 pode divergir se necessário (loss_task domina)

**Protocolo:**
1. Treinar Stage 2 do zero (ImageNet init)
2. Adicionar loss_distill durante treinamento
3. λ schedule: 1.0 → 0.1 (anneal ao longo de épocas)

**Potencial de Ganho:** +2 a +5pp Stage 2 F1 (37% → 39-42%)  
**Esforço:** Médio (3-4 dias)  
**Risco:** Médio (pode não superar train from scratch puro)  
**Fundamentação:** 
- Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
- Romero et al. (2015) - "FitNets": distillation de features intermediárias > logits

**Documentação:** `docs_v6/11_knowledge_distillation_stage2.md`

---

## 🟢 **EXPLORATÓRIA** - Arquiteturas Alternativas

### **Exp 12A: Transformer-Based Backbone**

**Objetivo:** Vision Transformer (ViT) pode capturar padrões geométricos melhor que CNNs.

**Motivação:**
- Partições AV1 são padrões **geométricos** (horizontal/vertical/quad/asymmetric)
- CNNs têm viés indutivo para texturas locais
- Transformers capturam dependências de longo alcance (global patterns)

**Arquitetura Proposta:**

```
Input: 16×16 block
    ↓
Patch Embedding (4×4 patches → 16 patches de 4×4)
    ↓
ViT-Small (6 layers, 8 heads, dim=384)
    ↓
[CLS] token → MLP Head → 10 classes
```

**Implementação:**

```python
from transformers import ViTModel, ViTConfig

config = ViTConfig(
    image_size=16,
    patch_size=4,
    num_channels=1,  # Grayscale
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=8
)

model = ViTModel(config)
```

**Desafios:**
- Dataset pequeno (152k samples) - ViT precisa de grandes datasets
- Solução: Heavy augmentation + regularização (dropout=0.3)

**Potencial de Ganho:** +3 a +8pp (mudança arquitetural radical)  
**Esforço:** Muito Alto (2 semanas)  
**Risco:** Alto (pode não convergir, ViT sensível a hiperparâmetros)  
**Fundamentação:** 
- Dosovitskiy et al. (2021) - "An Image is Worth 16x16 Words": ViT supera CNNs em fine-grained tasks
- Steiner et al. (2021) - "How to train your ViT": guidelines para datasets pequenos

**Custo:** Retreinar tudo do zero

**Documentação:** `docs_v6/12_transformer_backbone.md`

---

### **Exp 12B: Multi-Task Learning (Stage 2 prediz 3-way + 9-way simultaneamente)**

**Objetivo:** Stage 2 aprende hierarquia (SPLIT/RECT/AB) e classes finais (9 classes) juntas.

**Arquitetura:**

```
Input: 16×16 block
    ↓
Backbone (ResNet-18)
    ↓
Features (512-dim)
    ↓
    ├─ Head1 (3-way): SPLIT | RECT | AB
    └─ Head2 (9-way): HORZ | VERT | SPLIT | HORZ_A/B | VERT_A/B | HORZ_4 | VERT_4

Loss = α * Loss_3way + (1-α) * Loss_9way  (α=0.5)
```

**Vantagens:**
- ✅ Features otimizadas para **ambas** tarefas (hierárquica e flat)
- ✅ Se Head2 funcionar bem, **elimina Stage 3** (pipeline de 2 stages)
- ✅ Multi-task learning melhora generalização (Caruana, 1997)

**Implementação:**

```python
class MultiTaskStage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18()
        self.head_3way = nn.Linear(512, 3)  # SPLIT/RECT/AB
        self.head_9way = nn.Linear(512, 9)  # Final classes
    
    def forward(self, x):
        features = self.backbone(x)
        logits_3way = self.head_3way(features)
        logits_9way = self.head_9way(features)
        return logits_3way, logits_9way
```

**Protocolo:**
1. Treinar com loss combinada
2. Avaliar Head2 standalone (ignorar Head1)
3. Se Head2 > 45% F1: substituir pipeline hierárquico

**Potencial de Ganho:** +2 a +6pp (features otimizadas)  
**Esforço:** Alto (1 semana)  
**Risco:** Alto (pode colapsar para classes majoritárias em 9-way)  
**Fundamentação:** 
- Caruana (1997) - "Multitask Learning": compartilhar representações melhora generalização
- Ruder (2017) - "An Overview of Multi-Task Learning": auxiliary tasks como regularização

**Documentação:** `docs_v6/12_multi_task_learning.md`

---

### **Exp 12C: End-to-End Differentiable Pipeline**

**Objetivo:** Treinar Stage 1 + Stage 2 + Stage 3 juntos com backpropagation (elimina erro cascata).

**Problema Atual:**
- Pipeline treinado stage-by-stage (não end-to-end)
- Erro cascata exponencial: Acc = S1 × S2 × S3

**Solução: Differentiable Routing com Gumbel-Softmax (Jang et al., 2017)**

**Arquitetura:**

```
Input → Stage 1 → Gumbel-Softmax (differentiable routing)
                  ↓
                  ├─ Route to NONE (output)
                  └─ Route to Stage 2 → Gumbel-Softmax
                                        ↓
                                        ├─ SPLIT (output)
                                        ├─ Route to Stage 3-RECT
                                        └─ Route to Stage 3-AB
```

**Gumbel-Softmax:**
```python
def gumbel_softmax_routing(logits, tau=1.0):
    # Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + gumbel_noise
    
    # Softmax with temperature
    soft_routing = F.softmax(y / tau, dim=-1)
    
    # Straight-through estimator (hard routing forward, soft gradient)
    hard_routing = F.one_hot(soft_routing.argmax(dim=-1), num_classes=logits.size(-1))
    routing = hard_routing - soft_routing.detach() + soft_routing
    
    return routing
```

**Training:**
```python
# Forward entire pipeline end-to-end
logits_s1 = stage1(x)
routing_s1 = gumbel_softmax_routing(logits_s1)  # (B, 2)

# Route to Stage 2 (differentiable)
x_stage2 = routing_s1[:, 1].unsqueeze(-1) * x  # Multiply by "PARTITION" prob
logits_s2 = stage2(x_stage2)
routing_s2 = gumbel_softmax_routing(logits_s2)  # (B, 3)

# Route to Stage 3 (omitted for brevity)
...

# Loss combines all stages
loss = CE(logits_s1, y_s1) + CE(logits_s2, y_s2) + CE(logits_s3, y_s3)
loss.backward()  # Gradient flows through entire pipeline!
```

**Potencial de Ganho:** +5 a +12pp (elimina erro cascata via joint optimization)  
**Esforço:** Extremo (1 mês - muito complexo)  
**Risco:** Extremo (routing pode colapsar, instabilidade de treinamento)  
**Fundamentação:** 
- Zhang et al. (2021) - "End-to-End Training of Multi-Stage Systems": joint > cascade
- Jang et al. (2017) - "Categorical Reparameterization with Gumbel-Softmax": diferenciável

**Limitação:** Pode não funcionar (routing pode colapsar para sempre NONE ou sempre PARTITION)

**Documentação:** `docs_v6/12_end_to_end_pipeline.md`

---

## 🔵 **ANÁLISE/DIAGNÓSTICO** - Entender Melhor o Problema

### **Exp 13A: Análise de Ativação e Feature Maps**

**Objetivo:** Visualizar o que Stage 1 vs Stage 2 aprenderam (explicar negative transfer).

**Técnicas:**

1. **Grad-CAM (Selvaraju et al., 2017):**
   - Visualizar regiões que Stage 1 e Stage 2 atendem
   - Hipótese: Stage 1 atende bloco inteiro (binary decision), Stage 2 atende bordas/geometria

2. **t-SNE Embeddings (Maaten & Hinton, 2008):**
   - Plotar features do backbone Stage 1 vs Stage 2 em 2D
   - Hipótese: Clusters diferentes, indicando feature incompatibility

3. **Layer-wise Feature Ablation:**
   - Congelar layer1, layer2, layer3, layer4 individualmente
   - Medir impacto no F1 Stage 2
   - Descobrir: "qual layer causa negative transfer?"

**Output:** Figuras para paper/tese mostrando incompatibilidade visual

**Potencial de Ganho:** +0pp (não melhora métrica)  
**Esforço:** Médio (1 semana)  
**Risco:** Baixo  
**Valor PhD:** ⭐⭐⭐⭐⭐ **Alto** - explica o "porquê" do negative transfer (essencial para tese)

**Documentação:** `docs_v6/13_feature_analysis.md`

---

### **Exp 13B: Oracle Experiment (Upper Bound)**

**Objetivo:** Qual é o **teto teórico** do pipeline hierárquico?

**Protocolo:**

1. **Substituir Stage 2 por Ground Truth:**
   - Modificar pipeline evaluation para usar GT labels ao invés de Stage 2 predictions
   - Stage 3 recebe roteamento perfeito

2. **Medir performance:**
   ```
   Accuracy_oracle = Acc_Stage1 × 1.0 (Stage 2 perfeito) × Acc_Stage3
   ```

3. **Comparar com baseline:**
   - Se Accuracy_oracle < 55%: Pipeline hierárquico tem limitação fundamental
   - Se Accuracy_oracle > 65%: Vale investir em melhorar Stage 2

**Potencial de Ganho:** Define upper bound (insight, não métrica)  
**Esforço:** Baixo (1 dia)  
**Risco:** Baixo  
**Valor PhD:** ⭐⭐⭐⭐ **Alto** - justifica mudança arquitetural se teto for baixo

**Documentação:** `docs_v6/13_oracle_experiment.md`

---

### **Exp 13C: Ablation Study Completo**

**Objetivo:** Quantificar contribuição de cada componente do modelo.

**Testes:**

| Componente | Baseline | Sem Componente | Δ F1 | Conclusão |
|------------|----------|----------------|------|-----------|
| SE-blocks | 46.51% | ? | ? | Útil ou não? |
| Spatial Attention | 46.51% | ? | ? | Útil ou não? |
| CB-Focal Loss | 46.51% | ? (CE simples) | ? | Quantificar ganho |
| ImageNet Pretrain | 46.51% | ? (random init) | ? | Quanto vale pretrain? |
| Balanced Sampler | 46.51% | ? (random sampler) | ? | Necessário? |
| Dropout (0.3) | 46.51% | ? (sem dropout) | ? | Previne overfitting? |

**Potencial de Ganho:** +0pp (identifica componentes inúteis para remover)  
**Esforço:** Alto (1 semana - múltiplos treinamentos)  
**Risco:** Baixo  
**Valor PhD:** ⭐⭐⭐ **Médio** - mostra rigor científico, mas não gera insight novo

**Documentação:** `docs_v6/13_ablation_study.md`

---

## 📊 **RANKING POR POTENCIAL × ESFORÇO**

| Experimento | Potencial (pp) | Esforço | Risco | Valor PhD | Recomendação |
|-------------|----------------|---------|-------|-----------|--------------|
| **10A: Recuperar S2 Frozen** | +0~+2 | ⏱️ Baixo (1h) | 🟢 Baixo | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ **FAZER AGORA** |
| **10B: Confusion-Based Noise** | +1.5~+3 | ⏱️ Médio (1-2d) | 🟢 Baixo | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ Após 10A |
| **10C: Train-with-Predictions** | +2~+4 | ⏱️ Médio (2-3d) | 🟡 Médio | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ Alternativa a 10B |
| **11A: Adapter Layers** | +5~+10 S2<br>(+2~+5 pipe) | ⏱️ Alto (1sem) | 🟡 Médio | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ **Potencial alto** |
| **10D: Ensemble AB Real** | +1~+2 | ⏱️ Baixo (1d) | 🟢 Baixo | ⭐⭐ | ⭐⭐⭐ Quick win |
| **13B: Oracle Experiment** | Insight | ⏱️ Baixo (1d) | 🟢 Baixo | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ Para tese |
| **12B: Multi-Task Learning** | +2~+6 | ⏱️ Alto (1sem) | 🔴 Alto | ⭐⭐⭐⭐ | ⭐⭐⭐ Se 10B falhar |
| **13A: Feature Analysis** | +0 (insight) | ⏱️ Médio (1sem) | 🟢 Baixo | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ Contribuição tese |
| **11C: Knowledge Distillation** | +2~+5 S2 | ⏱️ Médio (3-4d) | 🟡 Médio | ⭐⭐⭐ | ⭐⭐⭐ Alternativa a 11A |
| **11B: Meta-Learning** | +3~+8 S2 | ⏱️ Muito Alto (2sem) | 🔴 Alto | ⭐⭐⭐⭐ | ⭐⭐ Complexo |
| **12A: Transformer Backbone** | +3~+8 | ⏱️ Muito Alto (2sem) | 🔴 Alto | ⭐⭐⭐⭐ | ⭐⭐ Dataset pequeno |
| **13C: Ablation Study** | +0 (insight) | ⏱️ Alto (1sem) | 🟢 Baixo | ⭐⭐⭐ | ⭐⭐ Rigor científico |
| **12C: End-to-End Pipeline** | +5~+12 | ⏱️ Extremo (1mês) | 🔴 Extremo | ⭐⭐⭐⭐⭐ | ⭐ Última opção |

**Legenda:**
- **Potencial:** Ganho esperado em pontos percentuais (pp) de accuracy
- **Esforço:** Tempo estimado de implementação + treinamento
- **Risco:** Probabilidade de falha técnica
- **Valor PhD:** Contribuição científica para tese (insights, novelty, rigor)
- **Recomendação:** Prioridade geral (⭐⭐⭐⭐⭐ = máxima)

---

## 🎯 **SEQUÊNCIA RECOMENDADA**

### **Fase 1: Estabilização** (Meta: 48%+ accuracy)

**Objetivo:** Resolver bloqueios críticos e atingir meta mínima.

```
Semana 1:
├─ Exp 10A: Recuperar Stage 2 frozen (1 dia)
├─ Exp 10B: Confusion-based noise injection (3 dias)
└─ Exp 10D: Ensemble AB real (1 dia)

Ganho esperado: 45.86% → 48-50%
```

**Critério de Sucesso:** Accuracy ≥ 48.0%

**Se falhar:** Pular para Fase 2 (Exp 11A)

---

### **Fase 2: Breakthrough** (Meta: 52%+ accuracy)

**Objetivo:** Melhorar Stage 2 fundamentalmente (resolver gargalo).

```
Semana 2-3:
├─ Exp 11A: Adapter layers (1 semana)
├─ Exp 13B: Oracle experiment (1 dia)
└─ Decisão baseada em Oracle:
    ├─ Se Oracle > 65%: Continuar hierárquico (investir em Stage 2)
    └─ Se Oracle < 60%: Considerar Exp 12B (multi-task) ou flatten

Ganho esperado: 48-50% → 52-56%
```

**Critério de Sucesso:** Stage 2 F1 ≥ 55% E Accuracy ≥ 52%

**Se Exp 11A falhar:** Tentar Exp 11C (Knowledge Distillation)

---

### **Fase 3: Exploração Avançada** (SOMENTE se Fase 1-2 falharem)

**Objetivo:** Mudanças arquiteturais radicais.

```
Semana 4-6:
└─ Escolher 1 dos 3:
    ├─ Exp 12A: Transformer backbone (se dataset parecer pequeno, evitar)
    ├─ Exp 12B: Multi-task learning (se Oracle < 60%)
    └─ Exp 12C: End-to-end pipeline (última opção, alto risco)

Ganho esperado: 48-50% → 55-60% (se funcionar)
```

**Critério de Abandono:** Se nenhum experimento Fase 3 > 50%, considerar:
- Coletar mais dados
- Mudar para abordagem não-hierárquica (flat 10-way)
- Repensar problema (partition prediction pode ter teto intrínseco)

---

### **Fase Paralela: Documentação para Tese**

**Experimentos para rodar em paralelo (não bloqueiam pipeline principal):**

```
Qualquer momento:
├─ Exp 13A: Feature analysis (1 semana)
├─ Exp 13C: Ablation study (1 semana)
└─ Documentação de falhas (ULMFiT, Train-from-Scratch, etc.)

Objetivo: Gerar figuras, tabelas e insights para capítulos da tese
```

---

## 📚 **Referências Principais**

### Transfer Learning & Negative Transfer
1. Yosinski et al. (2014) - "How transferable are features in deep neural networks?"
2. Pan & Yang (2010) - "A Survey on Transfer Learning"
3. Kornblith et al. (2019) - "Do Better ImageNet Models Transfer Better?"

### Catastrophic Forgetting & Fine-tuning
4. Howard & Ruder (2018) - "Universal Language Model Fine-tuning" (ULMFiT)
5. Rebuffi et al. (2017) - "Learning multiple visual domains with residual adapters"
6. Houlsby et al. (2019) - "Parameter-Efficient Transfer Learning for NLP"

### Learning with Noisy Labels & Robustness
7. Natarajan et al. (2013) - "Learning with Noisy Labels"
8. Hendrycks et al. (2019) - "Using Pre-Training Can Improve Model Robustness"
9. Heigold et al. (2016) - "An ensemble of deep neural networks for object detection"

### Class Imbalance
10. Cui et al. (2019) - "Class-Balanced Loss Based on Effective Number of Samples"
11. Lin et al. (2017) - "Focal Loss for Dense Object Detection"

### Meta-Learning
12. Finn et al. (2017) - "Model-Agnostic Meta-Learning for Fast Adaptation"
13. Nichol et al. (2018) - "On First-Order Meta-Learning Algorithms" (Reptile)

### Knowledge Distillation
14. Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
15. Romero et al. (2015) - "FitNets: Hints for Thin Deep Nets"

### Multi-Task Learning
16. Caruana (1997) - "Multitask Learning"
17. Ruder (2017) - "An Overview of Multi-Task Learning in Deep Neural Networks"

### Vision Transformers
18. Dosovitskiy et al. (2021) - "An Image is Worth 16x16 Words: Transformers for Image Recognition"
19. Steiner et al. (2021) - "How to train your ViT? Data, Augmentation, and Regularization"

### Ensemble Methods
20. Dietterich (2000) - "Ensemble Methods in Machine Learning"

### Differentiable Routing
21. Jang et al. (2017) - "Categorical Reparameterization with Gumbel-Softmax"
22. Zhang et al. (2021) - "Rethinking Pre-training and Self-training"

### Interpretability
23. Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks"
24. Maaten & Hinton (2008) - "Visualizing Data using t-SNE"

---

## 📝 **Notas Finais**

### Decisão Crítica: Hierárquico vs Flat

**Se Oracle Experiment (Exp 13B) mostrar:**
- **Upper bound > 65%:** Continuar hierárquico, investir em melhorar Stage 2 (Exp 11A/B/C)
- **Upper bound < 60%:** Arquitetura hierárquica tem limitação fundamental
  - Considerar mudar para **flat 10-way** (eliminar Stage 3, predizer tudo em Stage 2)
  - Ou **multi-task** (Exp 12B)

### Princípios de Pesquisa PhD

1. **Sempre fundamentar em literatura:** Cada experimento deve citar 2-3 papers
2. **Documentar falhas:** Experimentos negativos são tão valiosos quanto positivos
3. **Análise quantitativa:** Não apenas "melhorou", mas "melhorou X% porque Y"
4. **Reproducibilidade:** Fixar seeds, salvar checkpoints, documentar hiperparâmetros
5. **Análise crítica:** Questionar resultados, identificar limitações, propor melhorias

### Critérios de Publicação

**Para paper de conferência (ex: CVPR, ICCV):**
- Accuracy ≥ 55% (SOTA ou próximo)
- Contribuição novel (ex: Adapter Layers para video codec, End-to-End cascade)
- Ablation study completo (Exp 13C)
- Visualizações (Exp 13A)

**Para tese de doutorado:**
- Documentação completa de todas tentativas (incluindo falhas)
- Análise teórica profunda (negative transfer, distribution shift, etc.)
- Comparação com baselines (v5, flatten, etc.)
- Discussão de limitações e trabalhos futuros

---

**Última Atualização:** 13 de outubro de 2025  
**Próxima Revisão:** Após conclusão de Exp 10A (Stage 2 frozen recovery)
