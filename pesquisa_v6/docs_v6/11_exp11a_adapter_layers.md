# Experimento 11A: Adapter Layers no Stage 2

**Data:** 14 de outubro de 2025  
**Branch:** `feat/exp11a-adapter-layers`  
**Status:** ✅ **CONCLUÍDO** - Resultados abaixo da meta  
**Prioridade:** 🔴 **CRÍTICA** - Contorna bloqueio do Stage 2

**Resultado Final:** F1=47.74% (época 1/50) - **0.78pp ABAIXO do frozen baseline (48.52%)**

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

## 5. Resultados Obtidos

### 5.1 Descoberta Crítica: Época 1 foi a Melhor

**⚠️ RESULTADO INESPERADO: Modelo degradou com treinamento**

| Época | Val F1 Macro | SPLIT | RECT | AB | Observação |
|-------|--------------|-------|------|-----|------------|
| **1** | **47.74%** 🏆 | 41.68% | 63.41% | 38.15% | **BEST - salvo em checkpoint** |
| 5 | 46.91% | - | - | - | -0.83pp |
| 10 | 38.26% | - | - | - | -9.48pp (colapso) |
| 20 | 39.85% | - | - | - | Estabilizou baixo |
| 50 | 39.93% | 31.50% | 46.66% | 41.62% | Final: -7.81pp vs best |

**Degradação:** Best (época 1): 47.74% → Final (época 50): 39.93% = **-7.82pp**

### 5.2 Métricas Detalhadas - Época 1 (Checkpoint Best)

**Performance Geral:**
```
Macro F1:        47.74%  ❌ (meta: ≥50%)
Weighted F1:     50.43%
Accuracy:        51.01%
Macro Precision: 47.24%
Macro Recall:    49.73%
```

**Performance por Classe:**

| Classe | Precision | Recall | F1 | Support | vs Frozen | Status |
|--------|-----------|--------|-----|---------|-----------|--------|
| **SPLIT** | 35.65% | 50.17% | **41.68%** | 5,962 | ? | ⚠️ Baixo |
| **RECT**  | 61.07% | 65.93% | **63.41%** | 17,765 | ? | ✅ Melhor |
| **AB**    | 45.01% | 33.10% | **38.15%** | 14,529 | ? | ⚠️ Baixo recall |

**Confusion Matrix (Época 1):**
```
                Predito
              SPLIT  RECT    AB     Total
Real SPLIT    2991   1539   1432   5962  (50.2% correto)
     RECT     1609  11713   4443  17765  (65.9% correto)
     AB       3791   5929   4809  14529  (33.1% correto)
```

**Análise dos Erros:**
- **SPLIT:** 50% correto, mas **26% confundido com RECT**, 24% com AB
- **RECT:** 66% correto (melhor classe), 9% confundido com SPLIT, 25% com AB
- **AB:** **Apenas 33% correto!** 26% confundido com SPLIT, **41% predito como RECT**

### 5.3 Comparação com Baselines

| Abordagem | Macro F1 | SPLIT | RECT | AB | Status |
|-----------|----------|-------|------|-----|--------|
| **Frozen (baseline)** | **48.52%** | ? | ? | ? | ✅ Superior |
| **Fine-tuning (Exp 10A)** | 25.90% | - | - | 0% | ❌ Colapsou |
| **Adapters (Exp 11A)** | **47.74%** | 41.68% | 63.41% | 38.15% | ❌ **-0.78pp** |

**Conclusão:** Adapters ficaram **0.78pp ABAIXO** do frozen baseline, **NÃO ATINGIRAM** meta de 50%.

### 5.4 Análise do Problema: Por que Época 1?

#### Hipótese 1: Inicialização Near-Zero Ideal
- Adapters começam como **funções identidade** (std=1e-3)
- Backbone Stage 1 já tem features adequadas (~48% F1)
- **Treinamento adicional afasta da configuração ótima**

#### Hipótese 2: Learning Rate Muito Alta (1e-4)
- Adequado para ImageNet fine-tuning
- **MAS MUITO ALTO** para adapters pequenos (64 dim bottleneck)
- Após época 1, adapters divergiram do ótimo local

#### Hipótese 3: Cosine Annealing Inadequado
- LR decaiu de 1e-4 → 1e-7 em 50 épocas
- Modelo não conseguiu "retornar" ao ótimo da época 1
- Early stopping agressivo teria ajudado

#### Hipótese 4: Capacidade Insuficiente
- Bottleneck=64 pode ser muito pequeno
- Ablation v3 (bottleneck=128) necessária para confirmar

### 5.5 Estatísticas Gerais (50 épocas)

```
Média F1:         40.43%
Desvio padrão:    2.80pp
Mínimo:           36.27% (época 46)
Máximo:           47.74% (época 1) 🏆
```

**Interpretação:** Modelo **nunca melhorou** após época 1, degradou progressivamente.

---

## 6. Análise Crítica e Lições Aprendidas

### 6.1 Por que Adapters Não Funcionaram?

#### ❌ Falha 1: Negative Transfer Não Resolvido
- Esperado: Adapters preservam Stage 1 features enquanto adaptam Stage 2
- **Realidade:** F1 = 47.74% (ainda abaixo de frozen 48.52%)
- **Problema:** Bottleneck pequeno (64) limita capacidade de adaptação

#### ❌ Falha 2: Treinamento Prejudica Performance
- Época 1: 47.74% (próximo do frozen)
- Época 50: 39.93% (-7.82pp)
- **Problema:** LR muito alta para parameter-efficient methods

#### ❌ Falha 3: Classe AB vs RECT Confusão
- 41% de AB preditos como RECT (mesmo na melhor época)
- Adapters não conseguem distinguir features sutis
- **Problema estrutural:** Dataset imbalance + features similares

### 6.2 Comparação com Literatura

| Método | Dataset | Performance | Params Treináveis | Resultado |
|--------|---------|-------------|-------------------|-----------|
| **Rebuffi et al. (2017)** | Visual Decathlon | 96.2% de full FT | 0.7% | ✅ Sucesso |
| **Houlsby et al. (2019)** | GLUE (NLP) | 97.8% de full FT | 2% | ✅ Sucesso |
| **Exp 11A (nosso)** | AV1 Partition | 98.4% de frozen | 2.51% | ❌ **Abaixo de frozen** |

**Diferença-chave:** 
- Rebuffi/Houlsby: Adapters em tarefas **diferentes mas relacionadas** (ImageNet → CIFAR, BERT → GLUE)
- Exp 11A: Adapters em **hierarquia de mesma tarefa** (binary → 3-way partition)
- **Conclusão:** Particionamento AV1 tem características únicas que não se adequam bem a adapters

### 6.3 Trade-off Eficiência vs Performance

**✅ Eficiência alcançada:**
- Trainable: 288k params (2.51%)
- Frozen: 11.2M params (97.49%)
- Tempo treino: ~25 minutos (50 épocas)

**❌ Performance insuficiente:**
- F1: 47.74% vs frozen 48.52% = **-0.78pp**
- Trade-off: **97.5% menos params para 0.78pp de perda**
- **Veredicto:** Não vale a pena (degradação mesmo que marginal)

### 6.4 Lições para a Tese

#### Lição 1: "Minimal Adaptation is Optimal"
> **Descoberta:** Adapters com inicialização near-zero funcionam melhor na **época 1** (sem treinamento extensivo). Treinamento adicional degrada performance.

**Implicação:** Parameter-efficient methods devem usar:
- Learning rates **muito menores** (5e-5 vs 1e-4)
- **Early stopping agressivo** (patience=3-5)
- **Checkpointing frequente** (salvar a cada época)

#### Lição 2: "Frozen Baseline é Competitivo"
> **Descoberta:** Frozen backbone (Stage 1) + cabeça treinável já alcança **48.52% F1** sem fine-tuning. Adicionar adapters complexos não melhora (47.74%).

**Implicação:** Para hierarquias de mesma tarefa, **transferência simples (frozen) pode ser suficiente**.

#### Lição 3: "Epoch 1 Best = Sinal de Problema"
> **Descoberta:** Quando melhor performance ocorre na primeira época, indica:
> 1. Learning rate muito alta
> 2. Modelo já próximo do ótimo (inicialização boa)
> 3. Tarefa não beneficia de treinamento adicional

**Implicação:** Implementar **validation após época 1** e comparar com épocas 5/10 antes de prosseguir.

---

## 7. Recomendações para Próximos Passos

### 7.1 Opção A: Ablation v3 - Aumentar Capacidade (Bottleneck=128)
**Motivação:** Testar se bottleneck=64 é insuficiente

**Protocolo:**
```bash
python3 pesquisa_v6/scripts/004_train_stage2_redesigned.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --stage1-model logs/v6_experiments/stage1_improved/stage1_model_best.pt \
  --output-dir logs/v6_experiments/stage2_adapters_v3 \
  --epochs 30 \
  --use-adapters \
  --adapter-bottleneck 128 \
  --lr-adapter 5e-5 \
  --lr-head 2e-4 \
  --patience 10
```

**Mudanças:**
- Bottleneck: 64 → **128** (4x mais params)
- LR adapter: 1e-4 → **5e-5** (mais conservador)
- Epochs: 50 → **30** (early stopping)
- Patience: 5 → **10** (mais tolerante)

**Esperado:**
- F1 ≥ 50% se capacidade era o problema
- Se F1 < 48.52%, confirma que adapters não adequados

**Custo:** ~30 minutos de treino

### 7.2 Opção B: Retrain com Ajustes Críticos (Bottleneck=64, LR Baixo)
**Motivação:** Talvez época 1 seja ótimo, mas LR alta destruiu

**Protocolo:**
```bash
python3 pesquisa_v6/scripts/004_train_stage2_redesigned.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --stage1-model logs/v6_experiments/stage1_improved/stage1_model_best.pt \
  --output-dir logs/v6_experiments/stage2_adapters_v2 \
  --epochs 10 \
  --use-adapters \
  --adapter-bottleneck 64 \
  --lr-adapter 1e-5 \
  --lr-head 1e-4 \
  --patience 3 \
  --save-every-epoch
```

**Mudanças:**
- LR adapter: 1e-4 → **1e-5** (10x menor!)
- Epochs: 50 → **10** (early stopping agressivo)
- `--save-every-epoch`: Salvar modelo a cada época (inspecionar evolução)

**Esperado:**
- Se época 1 ainda é melhor → Confirma near-zero init é ótimo
- Se melhora após 5-10 épocas → LR era o problema

**Custo:** ~10 minutos de treino

### 7.3 Opção C: Documentar Limitação e Avançar (Exp 13B: Meta-Learning) ⭐ **RECOMENDADO**
**Motivação:** Adapters mostraram limitação fundamental para hierarquia AV1

**Justificativa:**
1. Época 1 best indica: **inicialização near-zero já é ótima**
2. F1=47.74% < frozen 48.52% indica: **adapters não ajudam**
3. Ablations v2/v3 provavelmente **não resolverão** problema fundamental
4. **Tempo melhor investido** em abordagens radicalmente diferentes (Meta-Learning, Few-Shot)

**Ação:**
- ✅ Documentar Exp 11A como **limitação conhecida**
- ✅ Atualizar `Proximos_Exp.md` com status "Concluído (negativo)"
- ✅ Partir para **Exp 13B (Meta-Learning)** ou **Exp 14 (Curriculum Learning)**

**Ganho:** Não desperdiçar tempo em variações de método já testado

---

## 8. Artefatos e Reprodutibilidade

### 8.1 Checkpoints Salvos

| Arquivo | Época | F1 Macro | Tamanho | Path |
|---------|-------|----------|---------|------|
| `stage2_model_best.pt` | **1** | **47.74%** 🏆 | 47 MB | `logs/v6_experiments/stage2_adapters/` |
| `stage2_model_final.pt` | 50 | 39.93% | 131 MB | `logs/v6_experiments/stage2_adapters/` |
| `stage2_history.pt` | 1-50 | - | 9.2 KB | `logs/v6_experiments/stage2_adapters/` |
| `stage2_metrics.json` | - | Summary | 845 B | `logs/v6_experiments/stage2_adapters/` |

**⚠️ Importante:** Checkpoint best foi salvo na **época 1**, não época 50!

### 8.2 Comando de Treino Exato
```bash
python3 pesquisa_v6/scripts/004_train_stage2_redesigned.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --stage1-model logs/v6_experiments/stage1_improved/stage1_model_best.pt \
  --output-dir logs/v6_experiments/stage2_adapters \
  --epochs 50 \
  --batch-size 128 \
  --use-adapters \
  --adapter-bottleneck 64 \
  --adapter-dropout 0.1 \
  --lr-adapter 1e-4 \
  --lr-head 5e-4 \
  --patience 5 \
  --device cuda \
  --seed 42
```

**Timestamp:** 13 de outubro de 2025, 22:31:29 (best checkpoint)

### 8.3 Validação Standalone (Script 009)
```bash
# Para validar checkpoint isoladamente
python3 pesquisa_v6/scripts/009_analyze_stage2_confusion.py \
  --stage2-model logs/v6_experiments/stage2_adapters/stage2_model_best.pt \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --device cuda
```

**Saída Esperada:** F1=47.74%, confusion matrix igual à documentada

---

## 9. Referências Complementares

### Papers Citados
1. **Rebuffi et al. (2017)** - "Learning multiple visual domains with residual adapters"
2. **Houlsby et al. (2019)** - "Parameter-Efficient Transfer Learning for NLP"
3. **Howard & Ruder (2018)** - "Universal Language Model Fine-tuning for Text Classification" (ULMFiT)
4. **Finn et al. (2017)** - "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"

### Experimentos Relacionados
- **Exp 02** (doc `02_fundamentos_congelamento.md`): Congelamento backbone
- **Exp 05A** (doc `05_avaliacao_pipeline_completo.md`): Pipeline com frozen (Acc=45.86%)
- **Exp 10A** (doc `PROBLEMA_CRITICO_STAGE2.md`): Fine-tuning failure (F1: 48.52%→25.90%)

---

## 10. Checklist de Execução

### Fase 1: Implementação ✅
- [x] AdapterModule implementado (bottleneck + residual)
- [x] Stage2ModelWithAdapters implementado (4 adapters)
- [x] Script 004 modificado com flag `--use-adapters`
- [x] Forward pass testado (shapes corretos)
- [x] Parameter count verificado (2.51% trainable)

### Fase 2: Treino ✅
- [x] Dataset carregado (152.6k train, 38.3k val)
- [x] Training executado (50 épocas, ~25 min)
- [x] Checkpoint best salvo (época 1, F1=47.74%)
- [x] Checkpoint final salvo (época 50, F1=39.93%)
- [x] Métricas JSON gerado

### Fase 3: Análise ✅
- [x] Best epoch identificado (época 1, não 50!)
- [x] Confusion matrix extraída
- [x] Degradação documentada (-7.82pp)
- [x] Comparação com frozen baseline (F1: 47.74% vs 48.52%)

### Fase 4: Documentação ✅
- [x] Seção 5 (Resultados) atualizada com métricas reais
- [x] Seção 6 (Análise Crítica) com lições aprendidas
- [x] Seção 7 (Próximos Passos) com 3 opções (A/B/C)
- [x] Seção 8 (Artefatos) com checkpoints e comandos
- [x] Confusion matrix e tabelas formatadas
- [x] Hipóteses de falha documentadas

### Fase 5: Git e Repositório ⏳
- [ ] `git add docs_v6/11_exp11a_adapter_layers.md`
- [ ] `git commit -m "docs(exp11a): Document results..."`
- [ ] `git push origin feat/exp11a-adapter-layers`

### Fase 6: Validação Opcional ⏳
- [ ] Script 009 (validação standalone confusion matrix)
- [ ] Script 008 (pipeline evaluation completo)

### Fase 7: Decisão Estratégica ⏳
- [ ] Opção A: Ablation v3 (bottleneck=128)?
- [ ] Opção B: Retrain (LR=1e-5, early stop)?
- [ ] **Opção C: Avançar para Exp 13B (Meta-Learning)** ⭐

**Status Final:** ✅ **CONCLUÍDO com resultado NEGATIVO** - F1=47.74% (época 1) **abaixo de frozen baseline 48.52%**

---

**Última Atualização:** 14 de outubro de 2025 - Documentação completa pós-treino
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
