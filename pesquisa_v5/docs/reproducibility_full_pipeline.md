# Guia Completo de Reprodutibilidade - Pipeline CNN AV1

Este documento descreve, passo a passo, como reproduzir os experimentos realizados até a avaliação completa do pipeline hierárquico. As instruções assumem que você está na raiz do projeto (`CNN_AV1`) em um ambiente Linux e que possui acesso aos mesmos artefatos brutos (arquivos `partition_frame_*.txt`, vídeos YUV/Y4M e diretórios preparados abaixo).

> **Ajuste caminhos conforme seu ambiente.** Neste guia são utilizados os diretórios do laboratório do projeto:
> - Dados brutos e saídas intermediárias: `/home/chiarorosa/experimentos/uvg`
> - Vídeos originais: `/home/chiarorosa/videoset/ugc-yuv`

## 0. Ambiente e dependências

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Mantenha a *virtualenv* ativada para todas as etapas seguintes.

## 1. Preparação dos dados brutos (scripts 004–007)

### 1.1 Gerar planilhas de partição (004)
Transforma os arquivos `partition_frame_*.txt` em planilhas Excel com a estrutura esperada pelas próximas etapas.

```bash
python pesquisa/004_prepare_partition_data_v2.py /home/chiarorosa/experimentos/uvg/
```

**Entradas:** subpastas contendo `partition_frame_X.txt`.
**Saídas:** arquivos `*-intra-*.xlsx` na mesma raiz.

### 1.2 Extrair blocos da componente Y (005)
Lê cada planilha gerada na etapa anterior, localiza o frame correspondente nos vídeos e exporta blocos de tamanhos 64, 32, 16 e 8.

```bash
python pesquisa/005_rearrange_video_v2.py \
  /home/chiarorosa/experimentos/uvg \
  /home/chiarorosa/videoset/ugc-yuv \
  y4m \
  1920 \
  1080
```

**Entradas:** planilhas `*-intra-*.xlsx` e arquivos de vídeo (`.y4m`).
**Saídas:** diretório `intra_raw_blocks/` contendo arquivos `_intra_raw_XX.txt`.

### 1.3 Consolidar blocos por sequência (006)
Agrupa ou renomeia os arquivos `_intra_raw_XX.txt` em um único arquivo por sequência e tamanho de bloco.

```bash
python pesquisa/006_merge_sample_v2.py /home/chiarorosa/experimentos/uvg/intra_raw_blocks
```

**Entradas:** diretório `intra_raw_blocks/` gerado na etapa 1.2.
**Saídas:** arquivos `<sequencia>_sample_XX.txt` no mesmo diretório.

### 1.4 Gerar labels e QPs (007)
Lê as planilhas Excel e produz, para cada vídeo e tamanho de bloco, os arquivos de rótulos e QPs alinhados aos blocos brutos.

```bash
python pesquisa/007_generate_label_qp_v2.py /home/chiarorosa/experimentos/uvg
```

**Saídas:** subdiretórios `labels/` e `qps/` dentro de `/home/chiarorosa/experimentos/uvg` com arquivos `*_labels_XX_intra.txt` e `*_qps_XX_intra.txt`.

## 2. Preparação dos datasets hierárquicos (scripts 008–011)

### 2.1 Dataset hierárquico completo (008)
Compatibiliza blocos, labels e QPs em tensores PyTorch por bloco.

```bash
python3 pesquisa/008_prepare_hierarchical_dataset.py --base-dir /home/chiarorosa/experimentos/uvg --block-size 16 --output-dir pesquisa/v5_dataset
```

### 2.2 Dataset somente particionado (opcional para Stage2/Stage3)

```bash
python3 pesquisa/008_prepare_hierarchical_dataset.py --base-dir /home/chiarorosa/experimentos/uvg --block-size 16 --output-dir pesquisa/v5_dataset_partitioned --partitioned-only
```

### 2.3 Datasets dos especialistas Stage3 (011)

```bash
python3 pesquisa/011_prepare_stage3_datasets.py --block-size 16 --source-root pesquisa/v5_dataset_partitioned --output-root pesquisa/v5_dataset_stage3
```

## 3. Treinamento dos modelos (scripts 009–012)

### 3.1 Stage1 - Classificador binário
```bash
python3 pesquisa/009_train_stage1.py --block-size 16 --dataset-root pesquisa/v5_dataset --epochs 12 --batch-size 128 --focal-gamma 1.5 --use-sampler --checkpoint-path pesquisa/logs/v5_stage1/stage1_model_block16_sampler.pt
```

### 3.2 Stage2 - Macro classes
```bash
# Treino principal (15 épocas)
python3 pesquisa/010_train_stage2.py --block-size 16 --dataset-root pesquisa/v5_dataset_partitioned --stage1-checkpoint pesquisa/logs/v5_stage1/stage1_model_block16_sampler.pt --epochs 15 --label-smoothing 0.02 --lr 5e-4 --checkpoint-path pesquisa/logs/v5_stage2/stage2_model_block16_classweights.pt

# Snapshot na época 8 (melhor macro-F1)
python3 pesquisa/010_train_stage2.py --block-size 16 --dataset-root pesquisa/v5_dataset_partitioned --stage1-checkpoint pesquisa/logs/v5_stage1/stage1_model_block16_sampler.pt --epochs 8 --label-smoothing 0.02 --lr 5e-4 --checkpoint-path pesquisa/logs/v5_stage2/stage2_model_block16_classweights_ep8.pt
```
# Exportar apenas o state dict da época 8
```bash
python3 -c "import torch; ckpt = torch.load('pesquisa/logs/v5_stage2/stage2_model_block16_classweights_ep8.pt', map_location='cpu'); torch.save(ckpt['model_state'], 'pesquisa/logs/v5_stage2/stage2_state_dict_block16_ep8.pt')"

```

### 3.3 Stage3 - Especialistas
Todos reutilizam `pesquisa/logs/v5_stage2/stage2_state_dict_block16_ep8.pt`.

```bash
# RECT (época 13)
python3 pesquisa/012_train_stage3.py --head RECT --dataset-root pesquisa/v5_dataset_stage3 --block-size 16 --stage2-state pesquisa/logs/v5_stage2/stage2_state_dict_block16_ep8.pt --epochs 13 --batch-size 128 --checkpoint-path pesquisa/logs/v5_stage3/stage3_RECT_model_block16_ep13.pt

# 1TO4 (época 14)
python3 pesquisa/012_train_stage3.py --head 1TO4 --dataset-root pesquisa/v5_dataset_stage3 --block-size 16 --stage2-state pesquisa/logs/v5_stage2/stage2_state_dict_block16_ep8.pt --epochs 14 --batch-size 128 --checkpoint-path pesquisa/logs/v5_stage3/stage3_1TO4_model_block16_ep14.pt

# AB (configuração com melhores resultados atuais)
python3 pesquisa/012_train_stage3.py --head AB --dataset-root pesquisa/v5_dataset_stage3 --block-size 16 --stage2-state pesquisa/logs/v5_stage2/stage2_state_dict_block16_ep8.pt --epochs 9 --batch-size 64 --label-smoothing 0.0 --use-sampler --lr 1e-4 --checkpoint-path pesquisa/logs/v5_stage3/stage3_AB_model_block16_augstrong.pt
```

## 4. Avaliação do pipeline completo (013)

O script `013_run_pipeline_eval.py` monta o modelo hierárquico, executa inferência por lote e gera JSON/CSV com métricas detalhadas.

baseline:
```bash
python pesquisa/013_run_pipeline_eval.py --dataset-root pesquisa/v5_dataset --block-size 16 --split val --stage1-checkpoint pesquisa/logs/v5_stage1/stage1_model_block16_sampler.pt --stage2-checkpoint pesquisa/logs/v5_stage2/stage2_model_block16_classweights_ep8.pt --rect-checkpoint pesquisa/logs/v5_stage3/stage3_RECT_model_block16_ep13.pt --ab-checkpoint pesquisa/logs/v5_stage3/stage3_AB_model_block16_augstrong.pt --one2four-checkpoint pesquisa/logs/v5_stage3/stage3_1TO4_model_block16_ep14.pt --output-json pesquisa/logs/v5_pipeline_eval_val.json --csv-path pesquisa/logs/v5_pipeline_eval_val.csv
```

threshold 0.5
```bash
python pesquisa/013_run_pipeline_eval.py --dataset-root pesquisa/v5_dataset --block-size 16 --split val --stage1-threshold 0.5 --stage1-checkpoint pesquisa/logs/v5_stage1/stage1_model_block16_sampler.pt --stage2-checkpoint pesquisa/logs/v5_stage2/stage2_model_block16_classweights_ep8.pt --rect-checkpoint pesquisa/logs/v5_stage3/stage3_RECT_model_block16_ep13.pt --ab-checkpoint pesquisa/logs/v5_stage3/stage3_AB_model_block16_augstrong.pt --one2four-checkpoint pesquisa/logs/v5_stage3/stage3_1TO4_model_block16_ep14.pt --output-json pesquisa/logs/v5_pipeline_eval_val_th050.json --csv-path pesquisa/logs/v5_pipeline_eval_val_th050.csv
```

## 5. Artefatos finais e próximos passos

- **Checkpoints principais:**
  - Stage1: `pesquisa/logs/v5_stage1/stage1_model_block16_sampler.pt`
  - Stage2: `pesquisa/logs/v5_stage2/stage2_model_block16_classweights.pt`
  - Stage2 (época 8, state dict): `pesquisa/logs/v5_stage2/stage2_state_dict_block16_ep8.pt`
  - Stage3 RECT: `pesquisa/logs/v5_stage3/stage3_RECT_model_block16_ep13.pt`
  - Stage3 1TO4: `pesquisa/logs/v5_stage3/stage3_1TO4_model_block16_ep14.pt`
  - Stage3 AB: `pesquisa/logs/v5_stage3/stage3_AB_model_block16_augstrong.pt`

- **Relatórios:**
  - Métricas completas: `pesquisa/logs/v5_pipeline_eval_val_th050.json`
  - Predições detalhadas: `pesquisa/logs/v5_pipeline_eval_val_th050.csv`

- **Análises adicionais:**
  - Atualize `pesquisa/notebooks/pipeline_analysis.ipynb` para visualizar matrizes de confusão ou comparar limiares (`--stage1-threshold`).

Seguindo essas etapas, qualquer pesquisador terá a mesma base de dados processada, modelos treinados e relatórios de avaliação gerados, reproduzindo integralmente o pipeline hierárquico desenvolvido até o momento.
