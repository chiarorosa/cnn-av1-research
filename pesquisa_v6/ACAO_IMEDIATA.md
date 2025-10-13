# AÇÃO IMEDIATA: Resolver Erro em Cascata

**Data:** 13 de outubro de 2025  
**Status:** 🔴 **IMPLEMENTAR HOJE**  
**Objetivo:** Accuracy ≥ 48% (gap atual: -0.34pp)

---

## ❌ POR QUE NÃO FAZER MAIS DIAGNÓSTICOS?

**Resposta:** O problema JÁ está completamente diagnosticado em `docs_v6/`.

**Evidências Documentadas:**
- ✅ **doc 05**: Stage 3-RECT: 68.44% standalone → 4.49% pipeline (-93.4%)
- ✅ **doc 05**: Stage 3-AB: 24.50% standalone → 1.51% pipeline (-93.8%)
- ✅ **doc 05**: HORZ colapsou (0% F1), Stage 2 confunde RECT→AB sistematicamente
- ✅ **doc 08**: Hipótese H2.1 (distribution shift) **REJEITADA** experimentalmente

**Root Cause CONFIRMADO:**
> Stage 3-RECT e Stage 3-AB foram treinados apenas com samples CORRETOS. No pipeline, recebem samples ERRADOS do Stage 2 e colapsam.

**Não precisamos de:**
- ❌ Script 009 (analyze_stage2_confusion) → doc 05 já analisou
- ❌ Script 010 (diagnose_stage3_rect) → doc 05 já diagnosticou
- ❌ Mais análises de confusão → sabemos que RECT→AB e AB→HORZ_B

**Precisamos de:**
- ✅ **AÇÃO DIRETA:** Treinar Stage 3 robusto a erros do Stage 2

---

## 🎯 SOLUÇÃO: Noise Injection (3-4 dias)

### Fundamentação Teórica

**Técnica:** Adversarial Training / Noise Injection
- **Paper 1:** Hendrycks et al., 2019 - "Using Pre-Training Can Improve Model Robustness"
- **Paper 2:** Natarajan et al., 2013 - "Learning with Noisy Labels"
- **Paper 3:** Recht et al., 2019 - "Do ImageNet Classifiers Generalize to ImageNet?"

**Princípio:**
> Treinar Stage 3 com 20-30% "dirty samples" (erros simulados do Stage 2). Modelo aprende robustez à distribuição real do pipeline.

**Por que funciona:**
- Stage 3 aprende a **rejeitar ou corrigir** inputs errados do Stage 2
- Distribui probability mass melhor (não colapsa em classe default)
- Generaliza para distribuição real do pipeline (train-test distribution match)

---

## 📋 PLANO DE IMPLEMENTAÇÃO (3-4 dias)

### Dia 1 (HOJE - 13/10): Implementação 🔴

**1. Modificar Script 005 (Stage 3-RECT)** - 2-3 horas

```bash
cd /home/chiarorosa/CNN_AV1/pesquisa_v6/scripts
cp 005_train_stage3_rect.py 005_train_stage3_rect_robust.py
```

**Mudanças necessárias:**

```python
# Adicionar argumento
parser.add_argument('--noise-injection', type=float, default=0.0,
                    help='Fraction of noise samples (0.0-1.0)')
parser.add_argument('--noise-sources', nargs='+', 
                    choices=['AB', 'SPLIT'], default=['AB'])

# Modificar dataset loading
class NoisyDataset(Dataset):
    def __init__(self, clean_dataset, noise_datasets, noise_ratio=0.25):
        self.clean = clean_dataset
        self.noise_datasets = noise_datasets
        self.noise_ratio = noise_ratio
        
        # Calcular total samples com noise
        n_clean = int(len(clean_dataset) * (1 - noise_ratio))
        n_noise = len(clean_dataset) - n_clean
        
        self.indices_clean = list(range(n_clean))
        self.indices_noise = []
        
        # Distribuir noise entre sources
        per_source = n_noise // len(noise_datasets)
        for noise_ds in noise_datasets:
            noise_idx = np.random.choice(len(noise_ds), per_source, replace=False)
            self.indices_noise.extend([(noise_ds, idx) for idx in noise_idx])
    
    def __getitem__(self, idx):
        if idx < len(self.indices_clean):
            # Sample limpo
            return self.clean[self.indices_clean[idx]]
        else:
            # Sample ruidoso (label aleatório)
            noise_ds, noise_idx = self.indices_noise[idx - len(self.indices_clean)]
            x, _ = noise_ds[noise_idx]
            y_random = torch.randint(0, 2, (1,)).item()  # HORZ ou VERT aleatório
            return x, y_random
    
    def __len__(self):
        return len(self.clean)

# No main():
if args.noise_injection > 0:
    # Carregar noise sources
    noise_datasets = []
    if 'AB' in args.noise_sources:
        ab_data = torch.load('v6_dataset_stage3/AB/block_16/train.pt')
        noise_datasets.append(TensorDataset(ab_data['blocks'], ab_data['labels']))
    if 'SPLIT' in args.noise_sources:
        # Extrair SPLIT do dataset principal
        main_data = torch.load('v6_dataset/block_16/train.pt')
        split_mask = main_data['labels'] == 3
        split_blocks = main_data['blocks'][split_mask]
        split_labels = main_data['labels'][split_mask]
        noise_datasets.append(TensorDataset(split_blocks, split_labels))
    
    train_dataset = NoisyDataset(train_dataset_clean, noise_datasets, 
                                  noise_ratio=args.noise_injection)
```

**2. Modificar Script 006 (Stage 3-AB)** - 2-3 horas

Mesma lógica, mas noise sources = ['RECT', 'SPLIT'], 4 classes (HORZ_A/B, VERT_A/B).

---

### Dia 2-3 (14-15/10): Treinamento 🔴

**1. Treinar Stage 3-RECT Robust** - 1.5 dias

```bash
source .venv/bin/activate

python3 005_train_stage3_rect_robust.py \
  --dataset-dir v6_dataset_stage3/RECT/block_16 \
  --noise-injection 0.25 \
  --noise-sources AB SPLIT \
  --epochs 30 \
  --batch-size 128 \
  --lr 3e-4 \
  --output-dir logs/v6_experiments/stage3_rect_robust \
  --device cuda

# Tempo estimado: ~30 min/epoch × 30 epochs = 15h = 1.5 dias
```

**Métricas esperadas:**
- Standalone F1: 65-70% (pode cair um pouco vs 68.44%)
- **Pipeline F1: 15-30%** (vs 4.49% atual) ✅ **+234-568%**

**2. Treinar Stage 3-AB Robust** - 1.5 dias

```bash
python3 006_train_stage3_ab_fgvc_robust.py \
  --dataset-dir v6_dataset_stage3/AB/block_16 \
  --noise-injection 0.25 \
  --noise-sources RECT SPLIT \
  --epochs 30 \
  --batch-size 128 \
  --lr 3e-4 \
  --output-dir logs/v6_experiments/stage3_ab_robust \
  --device cuda

# Tempo estimado: ~40 min/epoch × 30 epochs = 20h = 1.5 dias
```

**Métricas esperadas:**
- Standalone F1: 20-25% (similar ao atual 24.50%)
- **Pipeline F1: 5-12%** (vs 1.51% atual) ✅ **+231-695%**

---

### Dia 4 (16/10): Avaliação 🔴

**1. Re-avaliar Pipeline** - 2 horas

```bash
python3 008_run_pipeline_eval_v6.py \
  --dataset-dir v6_dataset/block_16 \
  --stage1-model logs/v6_experiments/stage1/stage1_model_best.pt \
  --stage2-model logs/v6_experiments/stage2/stage2_model_best.pt \
  --stage3-rect-model logs/v6_experiments/stage3_rect_robust/stage3_rect_model_best.pt \
  --stage3-ab-models logs/v6_experiments/stage3_ab_robust/stage3_ab_model_best.pt \
                    logs/v6_experiments/stage3_ab/stage3_ab_model_best.pt \
                    logs/v6_experiments/stage3_ab/stage3_ab_model_best.pt \
  --threshold 0.45 \
  --device cuda \
  --output-dir logs/v6_experiments/pipeline_eval_robust
```

**Expectativa:**
- Accuracy atual: 47.66%
- **Accuracy com robust: 48.7-50.2%** (+1.0-2.5pp) ✅✅

**2. Análise de Resultados** - 2 horas

```bash
# Comparar:
# - Pipeline baseline (doc 05): Acc=47.66%
# - Pipeline robust (novo): Acc=48.7-50.2%?

# Métricas-chave:
# - HORZ F1: 0% → >5%?
# - VERT_A F1: 0% → >3%?
# - Stage 3-RECT pipeline acc: 4.49% → >15%?
# - Stage 3-AB pipeline acc: 1.51% → >5%?
```

**3. Documentação** - 2 horas

Criar `docs_v6/09_noise_injection_stage3.md`:
- Motivação (erro cascata)
- Metodologia (noise injection 25%)
- Resultados (comparação baseline vs robust)
- Análise (por que funcionou ou não)
- Próximos passos (se não atingir 48%)

---

## 🎯 Critérios de Sucesso

| Métrica | Baseline | Meta Robust | Mínimo Aceitável |
|---------|----------|-------------|------------------|
| **Pipeline Accuracy** | 47.66% | 48.7-50.2% | **≥48.0%** ✅ |
| Stage 3-RECT pipeline | 4.49% | 15-30% | ≥10% |
| Stage 3-AB pipeline | 1.51% | 5-12% | ≥3% |
| HORZ F1 | 0% | >5% | >2% |
| Classes colapsadas | 5 (HORZ, 3xAB) | 0-2 | ≤3 |

**Se atingir ≥48%:** ✅ **SUCESSO** → Documentar e finalizar

**Se 47.8-48%:** ⚠️ **PARCIAL** → Considerar técnica adicional (Focal Loss tuning)

**Se <47.8%:** ❌ **FALHA** → Analisar por que não funcionou, considerar outras técnicas

---

## 📊 Cronograma Detalhado

| Dia | Data | Atividade | Horas | Status |
|-----|------|-----------|-------|--------|
| **1** | **13/10** | Implementar noise injection (scripts 005, 006) | 6h | ⏳ **HOJE** |
| 2 | 14/10 | Treinar Stage 3-RECT robust (epochs 1-15) | 8h | ⏳ |
| 2 | 14/10 noite | Treinar Stage 3-RECT robust (epochs 16-30) | overnight | ⏳ |
| 3 | 15/10 | Treinar Stage 3-AB robust (epochs 1-15) | 8h | ⏳ |
| 3 | 15/10 noite | Treinar Stage 3-AB robust (epochs 16-30) | overnight | ⏳ |
| 4 | 16/10 | Re-avaliar pipeline + análise + documentação | 6h | ⏳ |

**Total:** 3-4 dias

---

## 🚀 COMEÇAR AGORA (HOJE - 13/10)

```bash
# 1. Abrir terminal
cd /home/chiarorosa/CNN_AV1/pesquisa_v6/scripts

# 2. Criar scripts robustos
cp 005_train_stage3_rect.py 005_train_stage3_rect_robust.py
cp 006_train_stage3_ab_fgvc.py 006_train_stage3_ab_fgvc_robust.py

# 3. Implementar noise injection (próximas 6 horas)
# - Adicionar argumento --noise-injection
# - Implementar NoisyDataset class
# - Carregar noise sources
# - Testar com toy dataset

# 4. Validar implementação (30 min)
python3 005_train_stage3_rect_robust.py \
  --dataset-dir v6_dataset_stage3/RECT/block_16 \
  --noise-injection 0.25 \
  --noise-sources AB \
  --epochs 1 \
  --batch-size 32 \
  --device cuda
# Verificar que treina sem erro

# 5. Lançar treinamento completo (amanhã cedo)
```

---

## 📚 Referências Científicas

1. **Hendrycks, D., Mazeika, M., Wilson, D., & Gimpel, K. (2019)**  
   *Using Pre-Training Can Improve Model Robustness and Uncertainty*  
   ICML 2019  
   → Mostra que treinar com noise melhora robustez a distribution shift

2. **Natarajan, N., Dhillon, I. S., Ravikumar, P. K., & Tewari, A. (2013)**  
   *Learning with Noisy Labels*  
   NeurIPS 2013  
   → Teoria de aprendizado com label corruption

3. **Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019)**  
   *Do ImageNet Classifiers Generalize to ImageNet?*  
   ICML 2019  
   → Distribution shift entre train e test causa degradação

4. **Shimodaira, H. (2000)**  
   *Improving predictive inference under covariate shift*  
   Journal of Statistical Planning and Inference  
   → Fundamentos teóricos de covariate shift

5. **Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010)**  
   *A theory of learning from different domains*  
   Machine Learning  
   → Domain adaptation e distribution mismatch

---

**Próxima ação:** Abrir `005_train_stage3_rect.py` e começar implementação noise injection! 🚀
