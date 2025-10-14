# Experimento 11A: Adapter Layers no Stage 2

**Data:** 14 de outubro de 2025  
**Branch:** `feat/exp11a-adapter-layers`  
**Status:** 🟡 EM EXECUÇÃO  
**Prioridade:** 🔴 **CRÍTICA** - Contorna bloqueio do Stage 2

---

## 1. Contexto e Motivação

### 1.1 Problema Crítico Identificado

**Exp 10A revelou fragilidade fundamental do fine-tuning hierárquico:**
- Stage 1 backbone é ESSENCIAL (+39.53pp F1 vs ImageNet-only) ✅
- Fine-tuning frozen funciona (F1=48.52%) ✅
- **MAS:** Checkpoint save/load inconsistente (F1 degrada para 25.90%) ❌
- Bloqueio persiste: Exp 10B/10C/10D/13B dependem de Stage 2 funcional

**Tentativas anteriores falharam:**
- ❌ Exp 03 (ULMFiT): Discriminative LR, gradual unfreezing → F1 colapso
- ❌ Exp 04 (Train from scratch): F1=8.99% (Stage 1 backbone essencial)
- ❌ Exp 10A (Frozen checkpoint): Bug save/load não-determinístico

### 1.2 Fundamentação Teórica

**Rebuffi et al. (2017) - "Learning multiple visual domains with residual adapters":**
> "Adapter modules permitem task-specific adaptation sem modificar features base. Preservam conhecimento original enquanto aprendem nova task."

**Características dos Adapters:**
1. **Parameter-efficient:** < 1% dos parâmetros totais
2. **Residual connection:** Skip connection preserva features originais
3. **Task-specific:** Cada task tem seus próprios adapters
4. **No forgetting:** Backbone permanece frozen

**Houlsby et al. (2019) - "Parameter-Efficient Transfer Learning for NLP":**
- Adapters atingem 95-99% da performance de full fine-tuning
- Com apenas 0.5-2% dos parâmetros treináveis
- Elimina catastrophic forgetting

### 1.3 Hipótese

> **H11A:** "Inserir Adapter Layers após cada bloco residual do ResNet-18 Stage 2 permitirá adaptação task-specific (3-way classification) sem degradar features Stage 1 (binary). Esperado: F1=50-55% (+1.5-6.5pp vs frozen 48.52%) com checkpoint confiável."

---

## 2. Objetivo do Experimento

**Primário:**
1. Implementar arquitetura Adapter Layers no Stage 2
2. Treinar adapters (congelando backbone Stage 1) até convergência
3. Validar F1 ≥ 50% (superior ao frozen 48.52%)
4. **Garantir checkpoint confiável** (implementar validação inline)

**Secundário:**
5. Comparar com baseline frozen (Exp 10A training: F1=48.52%)
6. Avaliar pipeline completo (esperado: accuracy 48-50%)
7. Documentar ablation study (com/sem adapters em diferentes layers)

---

## 3. Arquitetura Proposta

### 3.1 Adapter Module (Rebuffi et al., 2017)

```python
class AdapterModule(nn.Module):
    """
    Residual Adapter Layer (Rebuffi et al., 2017)
    
    Bottleneck architecture: in_dim → bottleneck → in_dim
    Residual skip connection preserva features originais
    """
    def __init__(self, in_dim: int, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(in_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, in_dim)
        
        # Inicialização near-zero (Houlsby et al., 2019)
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] feature map
        Returns:
            adapted: [B, C, H, W] adapted features
        """
        # Global average pooling para obter feature vector
        B, C, H, W = x.shape
        pooled = x.mean(dim=[2, 3])  # [B, C]
        
        # Adapter transformation
        adapter_output = self.down_proj(pooled)
        adapter_output = self.activation(adapter_output)
        adapter_output = self.dropout(adapter_output)
        adapter_output = self.up_proj(adapter_output)  # [B, C]
        
        # Residual connection + broadcast
        adapter_output = adapter_output.view(B, C, 1, 1)
        return x + adapter_output
```

### 3.2 Stage2Model com Adapters

**Modificação na arquitetura:**
```python
class Stage2ModelWithAdapters(nn.Module):
    def __init__(self, pretrained=True, bottleneck_dim=64):
        super().__init__()
        
        # Backbone ResNet-18 (Stage 1 features)
        self.backbone = ImprovedBackbone(pretrained=pretrained)
        
        # FREEZE backbone (preserva Stage 1 features)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Adapter modules (trainable)
        self.adapter_layer1 = AdapterModule(64, bottleneck_dim)
        self.adapter_layer2 = AdapterModule(128, bottleneck_dim)
        self.adapter_layer3 = AdapterModule(256, bottleneck_dim)
        self.adapter_layer4 = AdapterModule(512, bottleneck_dim)
        
        # Classification head (trainable)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 3)  # SPLIT, RECT, AB
        )
    
    def forward(self, x):
        # Feature extraction (frozen)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # Layer 1 + Adapter
        x = self.backbone.layer1(x)
        x = self.adapter_layer1(x)
        
        # Layer 2 + Adapter
        x = self.backbone.layer2(x)
        x = self.adapter_layer2(x)
        
        # Layer 3 + Adapter
        x = self.backbone.layer3(x)
        x = self.adapter_layer3(x)
        
        # Layer 4 + Adapter
        x = self.backbone.layer4(x)
        x = self.adapter_layer4(x)
        
        # Global pooling + head
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        
        return x
```

**Contagem de parâmetros:**
- Backbone (frozen): 11.2M parâmetros ❌ (não treináveis)
- Adapters (4 modules): ~50k parâmetros ✅ (treináveis)
- Head: ~1.5k parâmetros ✅ (treináveis)
- **Total trainable:** ~52k (0.46% do modelo total)

---

## 4. Protocolo Experimental

### 4.1 Hiperparâmetros

```python
# Training config
EPOCHS = 50
BATCH_SIZE = 128
BOTTLENECK_DIM = 64

# Optimizer (discriminative LR)
LR_ADAPTER = 1e-4  # Adapters
LR_HEAD = 5e-4     # Head (maior LR, aprende mais rápido)
WEIGHT_DECAY = 1e-4

# Loss
CB_FOCAL_GAMMA = 2.0
CB_FOCAL_BETA = 0.9999

# Scheduler
SCHEDULER = 'CosineAnnealingLR'
T_MAX = 50
ETA_MIN = 1e-6

# Regularization
DROPOUT_ADAPTER = 0.1
DROPOUT_HEAD = 0.5
```

### 4.2 Modificações no Script 004

**Novo argumento:**
```python
parser.add_argument('--use-adapters', action='store_true',
                   help='Use Adapter Layers instead of fine-tuning')
parser.add_argument('--adapter-bottleneck', type=int, default=64,
                   help='Bottleneck dimension for adapters')
```

**Lógica de treinamento:**
```python
if args.use_adapters:
    model = Stage2ModelWithAdapters(
        pretrained=True,
        bottleneck_dim=args.adapter_bottleneck
    )
    
    # Optimizer: apenas adapters + head
    trainable_params = [
        {'params': [p for n, p in model.named_parameters() 
                   if 'adapter' in n], 'lr': args.lr_adapter},
        {'params': model.head.parameters(), 'lr': args.lr_head}
    ]
    optimizer = AdamW(trainable_params, weight_decay=args.weight_decay)
    
    print(f"🔧 Using Adapter Layers:")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Frozen params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
else:
    # Baseline: frozen backbone
    model = Stage2Model(pretrained=True)
    # ... (lógica original)
```

### 4.3 Checkpoint Validation (Lição do Exp 10A)

**Protocolo corrigido:**
```python
def save_and_validate_checkpoint(model, val_loader, criterion, device, path, expected_f1):
    """
    Salva checkpoint E valida imediatamente (evita bug do Exp 10A)
    """
    # 1. Salvar
    torch.save({
        'model_state_dict': model.state_dict(),
        'expected_f1': expected_f1,
        ...
    }, path)
    
    # 2. Carregar em modelo novo
    test_model = Stage2ModelWithAdapters(...)
    checkpoint = torch.load(path)
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model = test_model.to(device)
    
    # 3. Validar em subset (100 batches = ~25k samples)
    quick_metrics = validate_epoch(test_model, val_loader, criterion, device, max_batches=100)
    
    # 4. Verificar consistência
    delta_f1 = abs(quick_metrics['f1'] - expected_f1)
    if delta_f1 > 1.0:
        raise RuntimeError(f"⚠️ CHECKPOINT CORRUPTED: Expected F1={expected_f1:.2%}, got {quick_metrics['f1']:.2%}")
    
    print(f"✅ Checkpoint validated: F1={quick_metrics['f1']:.2%} (delta={delta_f1:.2%})")
    return quick_metrics
```

### 4.4 Comando de Execução

```bash
python3 pesquisa_v6/scripts/004_train_stage2_redesigned.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --epochs 50 \
  --batch-size 128 \
  --output-dir pesquisa_v6/logs/v6_experiments/stage2_adapters \
  --device cuda \
  --use-adapters \
  --adapter-bottleneck 64 \
  --lr-adapter 1e-4 \
  --lr-head 5e-4 \
  --stage1-model pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt \
  --seed 42
```

---

## 5. Resultados Esperados

### 5.1 Métricas de Sucesso

| Métrica | Baseline (Frozen) | Esperado (Adapters) | Ganho Mínimo |
|---------|-------------------|---------------------|--------------|
| **Val F1 Macro** | 48.52% | **≥ 50%** | +1.5pp |
| **Val Accuracy** | 51.19% | **≥ 53%** | +1.8pp |
| **F1 SPLIT** | 41.68% | ≥ 45% | +3.3pp |
| **F1 RECT** | 62.14% | ≥ 63% | +0.9pp |
| **F1 AB** | 41.73% | ≥ 45% | +3.3pp |

**Critério de sucesso:**
- ✅ F1 macro ≥ 50% (superior ao frozen)
- ✅ Checkpoint confiável (delta < 1pp após reload)
- ✅ Todas 3 classes funcionais (F1 > 40%)

### 5.2 Comparação com Literatura

**Rebuffi et al. (2017) - Visual Decathlon:**
- Adapters: 96.2% da performance de full fine-tuning
- Com apenas 0.7% parâmetros treináveis

**Houlsby et al. (2019) - GLUE NLP:**
- Adapters: 97.8% da performance de full fine-tuning
- Com 2% parâmetros treináveis

**Nossa expectativa:**
- Adapters: ≥ 98% da performance teórica de full fine-tuning (F1=50% vs teórico ~51%)
- Com 0.46% parâmetros treináveis

---

## 6. Ablation Study

### 6.1 Variações a Testar

**Exp 11A-v1 (Baseline):**
- Adapters em todos 4 layers (layer1-4)
- Bottleneck dim = 64

**Exp 11A-v2 (Adapters apenas em layers finais):**
- Adapters em layer3-4 apenas
- Hipótese: Layers iniciais (low-level features) não precisam adaptar

**Exp 11A-v3 (Bottleneck maior):**
- Bottleneck dim = 128
- Mais capacidade, mas risco de overfitting

**Exp 11A-v4 (Sem adapter em layer1):**
- Adapters em layer2-4
- Hipótese: Layer1 features são muito genéricas (edges, textures)

### 6.2 Protocolo de Ablation

1. Treinar cada variação (epochs=50)
2. Comparar F1 macro no validation set
3. Selecionar melhor configuração
4. Retreinar com 5 seeds diferentes (verificar estabilidade)

---

## 7. Riscos e Mitigações

### Risco 1: Adapters podem não ter capacidade suficiente

**Probabilidade:** Média (30%)  
**Sintoma:** F1 estagna em ~45-47% (não melhora vs frozen)  
**Mitigação:** 
- Aumentar bottleneck_dim (64 → 128 → 256)
- Testar Adapter em cada sub-bloco (não apenas após layer)

### Risco 2: Overfitting (modelo muito pequeno)

**Probabilidade:** Baixa (15%)  
**Sintoma:** Train F1 alto (~60%), Val F1 baixo (~40%)  
**Mitigação:**
- Aumentar dropout (0.1 → 0.2)
- Adicionar weight decay (1e-4 → 1e-3)
- Data augmentation mais agressivo

### Risco 3: Checkpoint bug persiste

**Probabilidade:** Muito Baixa (5%)  
**Sintoma:** F1 degrada após reload (mesmo com validação)  
**Mitigação:**
- Implementar checkpoint validation inline (descrito em 4.3)
- Se persistir: Issue fundamental no PyTorch, reportar bug

---

## 8. Pipeline Evaluation

### 8.1 Após Stage 2 com Adapters Funcionar

**Comando:**
```bash
python3 pesquisa_v6/scripts/008_run_pipeline_eval_v6.py \
  --stage1-model pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt \
  --stage2-model pesquisa_v6/logs/v6_experiments/stage2_adapters/stage2_model_best.pt \
  --stage3-rect-model pesquisa_v6/logs/test_noise_injection/stage3_rect_robust.pt \
  --stage3-ab-models \
      pesquisa_v6/logs/test_noise_injection_ab/stage3_ab_ensemble_model1.pt \
      pesquisa_v6/logs/test_noise_injection_ab/stage3_ab_ensemble_model2.pt \
      pesquisa_v6/logs/test_noise_injection_ab/stage3_ab_ensemble_model3.pt \
  --output-dir pesquisa_v6/logs/v6_experiments/pipeline_eval_adapters \
  --device cuda
```

**Expectativa:**
- Baseline (Exp 09 com Stage 2 colapsado): 45.86%
- Com Stage 2 adapters (F1=50%): **48-50% accuracy** (+2-4pp)

### 8.2 Análise de Cascade Error

**Esperado:** Redução significativa do cascade error

| Specialist | Standalone | Pipeline (Exp 09) | Pipeline (Exp 11A) | Melhoria |
|------------|------------|-------------------|-------------------|----------|
| RECT | 68% | 4% | **40-50%** | +36-46pp |
| AB | 24% | 1.5% | **12-18%** | +10-16pp |

**Razão:** Stage 2 com adapters distribui melhor samples para Stage 3 (não colapsa para RECT 100%)

---

## 9. Contribuição para a Tese

### 9.1 Contribuição Técnica

> "Demonstramos que Residual Adapters (Rebuffi et al., 2017) resolvem negative transfer em hierarchical video codec prediction. Com apenas 0.46% parâmetros treináveis, adapters preservam Stage 1 features (binary NONE vs PARTITION) enquanto adaptam para Stage 2 (3-way SPLIT/RECT/AB), superando fine-tuning frozen em +1.5-6.5pp F1."

### 9.2 Capítulos da Tese Impactados

**Capítulo 3 - Arquitetura:**
- Seção 3.5: "Parameter-Efficient Transfer Learning via Adapters"
- Comparação: Fine-tuning vs Frozen vs Adapters

**Capítulo 4 - Experimentos:**
- Seção 4.4: "Ablation Study - Adapter Configuration"
- Análise: Bottleneck dimension, layer placement

**Capítulo 5 - Resultados:**
- Seção 5.2: "Resolução do Negative Transfer Problem"
- Tabela comparativa: ULMFiT vs Frozen vs Adapters

---

## 10. Checklist de Execução

- [ ] **Fase 1: Implementação**
  - [ ] Criar `AdapterModule` em `v6_pipeline/models.py`
  - [ ] Criar `Stage2ModelWithAdapters` em `v6_pipeline/models.py`
  - [ ] Modificar Script 004 (adicionar `--use-adapters`)
  - [ ] Implementar checkpoint validation inline
  - [ ] Testar forward pass (verificar shapes)

- [ ] **Fase 2: Treinamento Baseline (v1)**
  - [ ] Executar treinamento (epochs=50, bottleneck=64)
  - [ ] Monitorar convergência (esperado: F1 ≥ 50% em ~30-40 epochs)
  - [ ] Validar checkpoint após save
  - [ ] Registrar métricas (F1, accuracy, per-class)

- [ ] **Fase 3: Validação Standalone**
  - [ ] Executar Script 009 (confusion matrix)
  - [ ] Verificar distribuição de predições (não colapso)
  - [ ] Comparar com baseline frozen (48.52%)

- [ ] **Fase 4: Pipeline Evaluation**
  - [ ] Executar Script 008 (pipeline completo)
  - [ ] Comparar com Exp 09 baseline (45.86%)
  - [ ] Analisar cascade error (RECT/AB degradation)

- [ ] **Fase 5: Ablation Study**
  - [ ] Treinar variação v2 (adapters apenas layer3-4)
  - [ ] Treinar variação v3 (bottleneck=128)
  - [ ] Treinar variação v4 (sem adapter em layer1)
  - [ ] Selecionar melhor configuração

- [ ] **Fase 6: Documentação**
  - [ ] Atualizar este documento com resultados
  - [ ] Criar visualizações (learning curves, confusion matrix)
  - [ ] Escrever seção na tese (draft)
  - [ ] Commit e push

---

## 11. Referências

1. **Rebuffi et al. (2017)** - "Learning multiple visual domains with residual adapters" - CVPR 2017
2. **Houlsby et al. (2019)** - "Parameter-Efficient Transfer Learning for NLP" - ICML 2019
3. Finn et al. (2017) - "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
4. Howard & Ruder (2018) - "Universal Language Model Fine-tuning" (ULMFiT)
5. Yosinski et al. (2014) - "How transferable are features in deep neural networks?"
6. Kornblith et al. (2019) - "Do Better ImageNet Models Transfer Better?"

---

**Status Atual:** 🟡 Fase 1 (Implementação) iniciada  
**Próxima Ação:** Implementar `AdapterModule` e `Stage2ModelWithAdapters`  
**Tempo Estimado:** 2-3 dias (implementação + treinamento + validação + ablation)  
**Data Atualização:** 14 de outubro de 2025, 16:15 BRT
