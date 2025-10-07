# Mudanças no Script 013_run_pipeline_eval.py

## Resumo
Tornei todos os checkpoints de especialistas (RECT, AB, 1TO4) **opcionais** para permitir avaliação do pipeline mesmo quando alguns especialistas não foram treinados (ex: 1TO4 sem dados).

## Mudanças Implementadas

### 1. Função `build_pipeline_model()`
- **Antes**: `rect_ckpt`, `ab_ckpt` eram `Path` obrigatórios
- **Depois**: São `Optional[Path]` e só carregam se fornecidos

```python
if rect_ckpt is not None:
    _load_state_filtered(model, rect_ckpt, prefix="specialist_heads.RECT")
if ab_ckpt is not None:
    _load_state_filtered(model, ab_ckpt, prefix="specialist_heads.AB")
if one2four_ckpt is not None:
    _load_state_filtered(model, one2four_ckpt, prefix="specialist_heads.1TO4")
```

### 2. Função `run_pipeline()`
- Novo parâmetro: `available_specialists: Optional[List[str]]`
- Lógica adaptada para verificar se especialista está disponível antes de usá-lo
- **Fallback**: Se especialista não disponível, usa primeira opção do grupo

```python
elif macro_name in stage3_heads and macro_name in available_specialists:
    # Usa especialista
    ...
elif macro_name in stage3_heads and macro_name not in available_specialists:
    # Fallback para primeira opção
    final_name = stage3_heads[macro_name][0]
    stage3_name = f"{macro_name}_FALLBACK"
```

### 3. Argumentos CLI
- `--rect-checkpoint`: Agora **opcional** (default=None)
- `--ab-checkpoint`: Agora **opcional** (default=None)  
- `--one2four-checkpoint`: Já era opcional

### 4. Função `main()`
- Detecta automaticamente quais especialistas estão disponíveis
- Imprime lista de especialistas disponíveis
- Adiciona `available_specialists` ao relatório JSON

## Comportamento

### Com todos especialistas:
```bash
python pesquisa/013_run_pipeline_eval.py \
  --rect-checkpoint ... \
  --ab-checkpoint ... \
  --one2four-checkpoint ...
```
→ Usa todos os especialistas normalmente

### Sem 1TO4 (caso atual):
```bash
python pesquisa/013_run_pipeline_eval.py \
  --rect-checkpoint ... \
  --ab-checkpoint ...
```
→ Usa RECT e AB, ignora 1TO4 (sem erro)

### Sem nenhum especialista:
```bash
python pesquisa/013_run_pipeline_eval.py \
  --stage1-checkpoint ... \
  --stage2-checkpoint ...
```
→ Usa apenas Stage1 + Stage2, fallback para Stage3

## Impacto nos Resultados

- **Accuracy final**: Pode ser menor quando faltam especialistas (usa fallback)
- **Métricas Stage3**: Só calculadas para especialistas disponíveis
- **CSV**: Marca predições com `_FALLBACK` quando especialista ausente
- **JSON**: Inclui lista de especialistas usados
