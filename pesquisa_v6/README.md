# Pipeline V6 - Guia de Uso

Pipeline hierárquico de deep learning para particionamento de blocos AV1.

## 📋 Pré-requisitos

```bash
# Ativar ambiente virtual
source .venv/bin/activate
```

---

## 🔄 Pipeline de Execução

### 1. Preparação de Dados

#### Script 001: Preparar Dataset Principal
```bash
python3 pesquisa_v6/scripts/001_prepare_v6_dataset.py \
  --base-path /home/chiarorosa/experimentos/uvg/ \
  --block-size 16 \
  --test-ratio 0.2 \
  --seed 42
```

**Saída:**
- `pesquisa_v6/v6_dataset/block_16/train.pt`
- `pesquisa_v6/v6_dataset/block_16/val.pt`
- `pesquisa_v6/v6_dataset/block_16/metadata.json`

---

#### Script 002: Preparar Datasets Stage 3
```bash
python3 pesquisa_v6/scripts/002_prepare_v6_stage3_datasets.py \
  --base-path /home/chiarorosa/experimentos/uvg/ \
  --block-size 16 \
  --test-ratio 0.2 \
  --seed 42
```

**Saída:**
- `pesquisa_v6/v6_dataset_stage3/RECT/block_16/train.pt`
- `pesquisa_v6/v6_dataset_stage3/RECT/block_16/val.pt`
- `pesquisa_v6/v6_dataset_stage3/AB/block_16/train_v1.pt` (ensemble)
- `pesquisa_v6/v6_dataset_stage3/AB/block_16/train_v2.pt` (ensemble)
- `pesquisa_v6/v6_dataset_stage3/AB/block_16/train_v3.pt` (ensemble)
- `pesquisa_v6/v6_dataset_stage3/AB/block_16/val.pt`

---

### 2. Treinamento

#### Script 003: Treinar Stage 1 (Binary: NONE vs PARTITION)
```bash
python3 pesquisa_v6/scripts/003_train_stage1_improved.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --output-dir pesquisa_v6/logs/v6_experiments/stage1 \
  --epochs 20 \
  --batch-size 128 \
  --lr 1e-3 \
  --focal-gamma 2.5 \
  --focal-alpha 0.25 \
  --use-hard-mining \
  --device cuda \
  --num-workers 4 \
  --seed 42
```

**Saída:**
- `pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt`
- `pesquisa_v6/logs/v6_experiments/stage1/stage1_model_final.pt`
- `pesquisa_v6/logs/v6_experiments/stage1/stage1_history.pt`
- `pesquisa_v6/logs/v6_experiments/stage1/stage1_metrics.json`

---

#### Script 004: Treinar Stage 2 (3-way: SPLIT, RECT, AB)
```bash
python3 pesquisa_v6/scripts/004_train_stage2_redesigned.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --stage1-model pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt \
  --output-dir pesquisa_v6/logs/v6_experiments/stage2 \
  --epochs 25 \
  --freeze-epochs 2 \
  --batch-size 128 \
  --lr 5e-4 \
  --lr-backbone 1e-5 \
  --focal-gamma 2.0 \
  --cb-beta 0.9999 \
  --label-smoothing 0.1 \
  --device cuda \
  --num-workers 4 \
  --seed 42
```

**Saída:**
- `pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt`
- `pesquisa_v6/logs/v6_experiments/stage2/stage2_model_final.pt`
- `pesquisa_v6/logs/v6_experiments/stage2/stage2_history.pt`
- `pesquisa_v6/logs/v6_experiments/stage2/stage2_metrics.json`

---

#### Script 005: Treinar Stage 3-RECT (Binary: HORZ vs VERT)
```bash
python3 pesquisa_v6/scripts/005_train_stage3_rect.py \
  --dataset-dir pesquisa_v6/v6_dataset_stage3/RECT/block_16 \
  --stage2-model pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt \
  --output-dir pesquisa_v6/logs/v6_experiments/stage3_rect \
  --epochs 30 \
  --batch-size 128 \
  --lr 3e-4 \
  --focal-gamma 2.0 \
  --focal-alpha 0.25 \
  --device cuda \
  --num-workers 4 \
  --seed 42
```

**Saída:**
- `pesquisa_v6/logs/v6_experiments/stage3_rect/stage3_rect_model_best.pt`
- `pesquisa_v6/logs/v6_experiments/stage3_rect/stage3_rect_model_final.pt`
- `pesquisa_v6/logs/v6_experiments/stage3_rect/stage3_rect_history.pt`
- `pesquisa_v6/logs/v6_experiments/stage3_rect/stage3_rect_metrics.json`

---

#### Script 006: Treinar Stage 3-AB FGVC (4-way fine-grained)
```bash
python3 pesquisa_v6/scripts/006_train_stage3_ab_fgvc.py \
  --dataset-dir pesquisa_v6/v6_dataset_stage3/AB/block_16 \
  --stage2-model pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt \
  --output-dir pesquisa_v6/logs/v6_experiments/stage3_ab \
  --epochs 30 \
  --batch-size 128 \
  --lr 3e-4 \
  --device cuda \
  --num-workers 4 \
  --seed 42
```

**Saída:**
- `pesquisa_v6/logs/v6_experiments/stage3_ab/stage3_ab_fgvc_best.pt` (F1=24.50%, 4/4 classes)
- `pesquisa_v6/logs/v6_experiments/stage3_ab/stage3_ab_fgvc_final.pt`
- `pesquisa_v6/logs/v6_experiments/stage3_ab/stage3_ab_fgvc_history.pt`
- `pesquisa_v6/logs/v6_experiments/stage3_ab/stage3_ab_fgvc_metrics.json`

**Nota:** Implementação FGVC com Two-Stage Fine-tuning, Center Loss, CutMix, Cosine Classifier.

**Modelo alternativo (referência):** `006_train_stage3_ab_ensemble_reference.py` (F1=10.35%, arquivado)

---

### 3. Otimização

#### Script 007: Otimizar Threshold Stage 1
```bash
python3 pesquisa_v6/scripts/007_optimize_thresholds.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --model-path pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt \
  --output-dir pesquisa_v6/logs/v6_experiments/threshold_optimization \
  --threshold-min 0.4 \
  --threshold-max 0.7 \
  --threshold-step 0.05 \
  --batch-size 256 \
  --device cuda \
  --num-workers 4
```

**Saída:**
- `pesquisa_v6/logs/v6_experiments/threshold_optimization/threshold_optimization_results.csv`
- `pesquisa_v6/logs/v6_experiments/threshold_optimization/threshold_optimization_summary.json`

**Resultados:**
- **Melhor F1**: threshold=0.45 → F1=72.79% (recomendado)
- **Melhor Precision**: threshold=0.70 → Precision=98.46%, F1=1.98%
- **Melhor Recall**: threshold=0.40 → Recall=83.12%, F1=69.29%
- **Threshold padrão (0.50)**: F1=72.27%, Precision=79.43%, Recall=66.30%

**Recomendação:** Usar threshold **0.45** para balanceamento ótimo F1.

---

### 4. Avaliação

#### Script 008: Avaliar Pipeline Completo
```bash
python3 pesquisa_v6/scripts/008_run_pipeline_eval_v6.py \
  --dataset-dir pesquisa_v6/v6_dataset/block_16 \
  --stage1-model pesquisa_v6/logs/v6_experiments/stage1/stage1_model_best.pt \
  --stage2-model pesquisa_v6/logs/v6_experiments/stage2/stage2_model_best.pt \
  --stage3-rect-model pesquisa_v6/logs/v6_experiments/stage3_rect/stage3_rect_model_best.pt \
  --stage3-ab-models \
    pesquisa_v6/logs/v6_experiments/stage3_ab/stage3_ab_model_v1_best.pt \
    pesquisa_v6/logs/v6_experiments/stage3_ab/stage3_ab_model_v2_best.pt \
    pesquisa_v6/logs/v6_experiments/stage3_ab/stage3_ab_model_v3_best.pt \
  --stage1-threshold 0.5 \
  --output-dir pesquisa_v6/logs/v6_experiments/pipeline_eval \
  --batch-size 256 \
  --device cuda \
  --num-workers 4
```

**Saída:**
- `pesquisa_v6/logs/v6_experiments/pipeline_eval/pipeline_metrics_val.json`
- `pesquisa_v6/logs/v6_experiments/pipeline_eval/pipeline_predictions_val.npz`
- `pesquisa_v6/logs/v6_experiments/pipeline_eval/pipeline_report_val.txt`

**Métricas:**
- Accuracy, Macro F1, Weighted F1
- Per-class F1 para todas as 8 classes (NONE, SPLIT, HORZ, VERT, HORZ_A, HORZ_B, VERT_A, VERT_B)
- Classification report completo
- Confusion matrix

---

## 🎯 Execução Rápida (Pipeline Completo)

```bash
# 1. Preparar dados
python3 pesquisa_v6/scripts/001_prepare_v6_dataset.py
python3 pesquisa_v6/scripts/002_prepare_v6_stage3_datasets.py

# 2. Treinar Stage 1
python3 pesquisa_v6/scripts/003_train_stage1_improved.py

# 3. Treinar Stage 2
python3 pesquisa_v6/scripts/004_train_stage2_redesigned.py

# 4. Treinar Stage 3-RECT
python3 pesquisa_v6/scripts/005_train_stage3_rect.py

# 5. Treinar Stage 3-AB FGVC
python3 pesquisa_v6/scripts/006_train_stage3_ab_fgvc.py

# 6. Otimizar Threshold Stage 1
python3 pesquisa_v6/scripts/007_optimize_thresholds.py

# 7. Avaliar Pipeline Completo
python3 pesquisa_v6/scripts/008_run_pipeline_eval_v6.py
```

---

## 📊 Monitoramento

Verificar métricas após treinamento:
```bash
# Stage 1
cat pesquisa_v6/logs/v6_experiments/stage1/stage1_metrics.json

# Stage 2
cat pesquisa_v6/logs/v6_experiments/stage2/stage2_metrics.json

# Stage 3-RECT
cat pesquisa_v6/logs/v6_experiments/stage3_rect/stage3_rect_metrics.json

# Stage 3-AB FGVC
cat pesquisa_v6/logs/v6_experiments/stage3_ab/stage3_ab_fgvc_metrics.json

# Threshold Optimization
cat pesquisa_v6/logs/v6_experiments/threshold_optimization/threshold_optimization_summary.json

# Pipeline Evaluation
cat pesquisa_v6/logs/v6_experiments/pipeline_eval/pipeline_metrics_val.json
```

---

## 🔧 Parâmetros Principais

### Dataset (001, 002)
- `--base-path`: Caminho para dataset raw
- `--block-size`: Tamanho do bloco (8, 16, 32, 64)
- `--test-ratio`: Proporção de validação (padrão: 0.2)
- `--seed`: Seed para reprodutibilidade

### Training (003, 004)
- `--dataset-dir`: Diretório do dataset processado
- `--output-dir`: Diretório de saída para logs e modelos
- `--epochs`: Número de épocas
- `--batch-size`: Tamanho do batch
- `--lr`: Taxa de aprendizado inicial
- `--device`: Dispositivo (cuda/cpu)
- `--num-workers`: Workers para DataLoader

### Losses
- `--focal-gamma`: Parâmetro γ do Focal Loss (Stage 1: 2.5, Stage 2: 2.0)
- `--focal-alpha`: Parâmetro α do Focal Loss (padrão: 0.25)
- `--cb-beta`: Beta para Class-Balanced Loss (padrão: 0.9999)
- `--label-smoothing`: Fator de suavização de labels (padrão: 0.1)

### Específicos
- `--use-hard-mining`: Habilitar Hard Negative Mining (Stage 1)
- `--freeze-epochs`: Épocas com backbone congelado (Stage 2)
- `--lr-backbone`: LR para backbone após descongelar (Stage 2)
- `--use-mixup`: Habilitar Mixup augmentation (Stage 3-AB)
- `--mixup-alpha`: Alpha do Mixup (padrão: 0.4)
- `--base-seed`: Seed base para ensemble (Stage 3-AB)

---

## 📈 Metas de Performance

| Stage | Métrica | v5 Atual | v6 Meta Fase 1 | v6 Resultado |
|-------|---------|----------|----------------|--------------|
| Stage 1 | F1 | 65.19% | 68-70% | **72.28%** ✅ |
| Stage 1 | Precision | 53.71% | 62-65% | **80.58%** ✅ |
| Stage 2 | Macro F1 | 33.41% | 45-50% | **47.58%** ✅ |
| Stage 3-RECT | F1 | 72.50% | 75-78% | **68.44%** ⚠️ |
| Stage 3-AB | F1 | 25.26% | 45-50% | **24.50%** ⚠️ |
| **Threshold Opt** | **Stage 1** | - | - | **0.45 (F1=72.79%)** ✅ |

**Status:** Scripts 001-007 completos. Próximo: Pipeline end-to-end (script 008).

---

## 🐛 Troubleshooting

**Dataset não encontrado:**
```bash
# Verificar caminho
ls -la /home/chiarorosa/experimentos/uvg/
```

**CUDA out of memory:**
```bash
# Reduzir batch size
--batch-size 64
```

**ImportError:**
```bash
# Verificar que está executando da raiz do projeto
pwd  # deve ser /home/chiarorosa/CNN_AV1
```
