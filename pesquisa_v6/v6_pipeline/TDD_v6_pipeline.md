# TDD – Pipeline CNN AV1 v6

## Objetivo do documento
- Consolidar a visão técnica dos componentes em `pesquisa_v6/v6_pipeline`.
- Explicar papéis, contratos de uso e dependências de cada script.
- Explicitar padrões de projeto aplicados para guiar manutenção e extensões.

## Visão geral da arquitetura
- A pipeline v6 mantém um fluxo hierárquico de três estágios (Stage 1 → Stage 2 → Stage 3) para reduzir desequilíbrio e especializar classificadores.
- Cada estágio combina: preparação dos dados (`data_hub.py`), transformações (`augmentation.py`), modelo (`models.py`), função de perda (`losses.py`) e monitoramento (`metrics.py`).
- `ensemble.py` adiciona uma camada de orquestração para Stage 3-AB, reduzindo variância com votações e meta-modelos.
- O diretório exporta apenas utilitários autocontidos; dependências externas vêm de PyTorch, NumPy, torchvision, scikit-learn e seaborn/matplotlib para relatórios.

## Referência de scripts

### `__init__.py`
- **Propósito:** Define metadados e os mapeamentos de rótulos usados em toda a pipeline. Oferece funções auxiliares `_label_to_stage{1,2,3}` para converter do rótulo cru (Stage 0) para as saídas esperadas em cada estágio.
- **Uso esperado:** Importado por consumidores que precisam de tabelas `PARTITION_*` ou convenções de Stage 2/3. Ex.: `from pesquisa_v6.v6_pipeline import PARTITION_ID_TO_NAME`.
- **Padrões de design:** atua como _module facade_, centralizando constantes e ajudando a manter coesão das regras de negócio de rótulos.

### `augmentation.py`
- **Propósito:** Reúne transformações de dados específicas por estágio, incluindo versões _label-aware_ para Stage 3-AB.
- **Principais componentes:**
  - Classes elementares (`HorizontalFlipWithLabelSwap`, `GaussianNoise`, etc.) implementam o protocolo `__call__`.
  - Pipelines (`Stage1Augmentation`, `Stage2Augmentation`, `Stage3RectAugmentation`, `Stage3ABAugmentation`) encadeiam transformações adequadas a cada fase.
  - `TestTimeAugmentation` provê TTA e agregação de previsões.
  - `get_augmentation(stage, train)` seleciona a estratégia correta.
- **Classes e uso:**
  - `HorizontalFlipWithLabelSwap`: espelha horizontalmente imagens Stage 3-AB e troca rótulos HORZ_A/HORZ_B.
  - `VerticalFlipWithLabelSwap`: espelha verticalmente e alterna rótulos VERT_A/VERT_B.
  - `Rotation90WithLabelRotate`: rotaciona 90°/270° e remapeia rótulos entre eixos horizontal/vertical.
  - `GaussianNoise`: injeta ruído gaussiano controlado por `sigma` para robustez a ruídos de captura.
  - `Cutout`: mascara uma região quadrada aleatória para incentivar invariância local.
  - `GridShuffle`: embaralha blocos de uma grade para aumentar diversidade estrutural.
  - `CoarseDropout`: remove múltiplos patches pequenos, atuando como regularização espacial.
  - `MixupAugmentation`: combina dois exemplos e retorna mistura + metadados de rótulo para uso com `MixupLoss`.
  - `Stage1Augmentation`: cadeia leve para Stage 1 com flips/rotações opcionais e ruído.
  - `Stage2Augmentation`: versão mais agressiva adicionando Cutout e GridShuffle.
  - `Stage3RectAugmentation`: ajustes moderados para separar HORZ vs VERT.
  - `Stage3ABAugmentation`: pipeline label-aware que atualiza rótulos após transformações geométricas.
  - `TestTimeAugmentation`: gera variantes fixas (original + flips + rotação) e agrega previsões de modelos.
- **Padrões de design:** 
  - _Factory Method_ em `get_augmentation`.
  - _Strategy_ no uso de objetos chamáveis por estágio; cada instância encapsula políticas de transformação intercambiáveis.
  - _Decorator_-like composição das transformações elementares dentro das pipelines.
- **Integrações:** Consumido por `data_hub.HierarchicalBlockDatasetV6` para aplicar augmentations em `__getitem__`. `Stage3ABAugmentation` também retorna rótulos ajustados.

### `data_hub.py`
- **Propósito:** Camada de acesso a dados e construção de datasets hierárquicos compatíveis com PyTorch.
- **Principais componentes:**
  - `BlockRecord` / `TorchBlockRecord` (_Value Objects_) guardam tensores e expõem `to_torch()`.
  - Funções de descoberta/carregamento (`index_sequences`, `load_block_records`, `train_test_split`) organizam arquivos brutos.
  - Mapas e funções `map_to_stage*` traduzem rótulos crus para cada estágio.
  - `HierarchicalBlockDatasetV6` implementa `torch.utils.data.Dataset` com suporte a augmentations label-aware.
  - Utilidades de balanceamento (`get_class_weights`, `create_balanced_sampler`, `create_ab_oversampled_dataset`) e filtros por estágio.
- **Classes e uso:**
  - `BlockRecord`: estrutura intermédia que armazena arrays NumPy e oferece inspeções rápidas antes da conversão para PyTorch.
  - `TorchBlockRecord`: versão Torch com tensores prontos para treinar; `samples`, `labels`, `qps`.
  - `HierarchicalBlockDatasetV6`: dataset PyTorch com acesso a `__getitem__` retornando dicionário de labels por estágio e aplicando augmentations conforme `stage`.
- **Padrões de design:** 
  - _Builder/Factory_ em `build_hierarchical_dataset_v6` para orquestrar conversões e injetar augmentations.
  - _Repository_ informal ao encapsular a leitura dos arquivos e expor `BlockRecord`.
- **Integrações:** Fornece datasets para _DataLoader_; augmentations de `augmentation.py` são injetadas; samplers podem ser plugados na etapa de treino.

### `ensemble.py`
- **Propósito:** Gerenciar ensembles do Stage 3-AB (votação, pesos e stacking).
- **Principais componentes:**
  - `ABEnsemble` (base) oferece `predict`, `predict_with_uncertainty`, persistência (`save_ensemble`, `load_ensemble`).
  - `WeightedEnsemble` especializa a previsão com votação ponderada.
  - `StackingEnsemble` adiciona um meta-modelo supervisionado sobre probabilidades base.
  - `create_ab_ensemble` instancia múltiplos modelos com seeds fixas.
  - `evaluate_ensemble_diversity` mede divergência/consenso entre membros.
- **Classes e uso:**
  - `ABEnsemble`: orquestra um conjunto de modelos básicos; fornece predição hard/soft, cálculo de incerteza e serialização dos pesos.
  - `WeightedEnsemble`: estende o ensemble base ponderando contribuições de cada modelo com pesos normalizados.
  - `StackingEnsemble`: guarda modelos base congelados e um meta-classificador treinável que consome probabilidades concatenadas.
- **Padrões de design:** 
  - _Template Method_ em `ABEnsemble.predict` (fluxo base reutilizado por subclasses).
  - _Strategy_ implícito na troca entre hard/soft voting e meta-modelagem.
  - _Factory Method_ em `create_ab_ensemble`; _Persistence Façade_ via `save_ensemble`/`load_ensemble`.
- **Integrações:** Trabalha sobre `models.Stage3ABModel` ou equivalentes; consome tensores já pré-processados pela etapa de dados.

### `losses.py`
- **Propósito:** Reunir perdas customizadas para lidar com desequilíbrio (Focal, Class-Balanced, Hard Negative Mining, Mixup).
- **Principais componentes:** Classes derivadas de `nn.Module` (`FocalLoss`, `ClassBalancedFocalLoss`, `HardNegativeMiningLoss`, `LabelSmoothingLoss`) e utilitário `MixupLoss`. `get_loss_function(stage, loss_config)` escolhe a perda adequada considerando hiperparâmetros.
- **Classes e uso:**
  - `FocalLoss`: reduz peso de amostras fáceis e é usada nos estágios Stage 1/3 para tratar desequilíbrio.
  - `ClassBalancedFocalLoss`: ajusta pesos com número efetivo de amostras; indicado para Stage 2 e Stage 3-AB ao configurar `samples_per_class`.
  - `MixupLoss`: provê métodos `mixup_data` e `mixup_criterion` para integrar mixup ao loop de treino.
  - `HardNegativeMiningLoss`: filtra negativos fáceis no Stage 1 quando `hard_mining` está habilitado.
  - `LabelSmoothingLoss`: alternativa para regularizar modelos multi-classes contra overconfidence.
- **Padrões de design:** 
  - _Strategy_ – cada loss encapsula uma política de otimização intercambiável.
  - _Factory Method_ em `get_loss_function` para injetar configurações de maneira declarativa.
- **Integrações:** Funções de treino escolhem a loss certo por estágio. `MixupLoss` pode ser combinado com augmentations para gerar batches virtuais.

### `metrics.py`
- **Propósito:** Calcular métricas (binárias e multi-classe), gerar matrizes de confusão, curvas PR e rastrear histórico de treino.
- **Principais componentes:** 
  - Funções `compute_metrics`, `compute_binary_metrics`, `compute_stage_metrics`, `find_optimal_threshold`, `plot_confusion_matrix`, `plot_precision_recall_curve`.
  - `MetricsTracker` armazena histórico e gera gráficos.
  - `print_metrics_summary` fornece um _console report_ padronizado.
- **Classes e uso:**
  - `MetricsTracker`: mantém listas de perdas/metricas por época, exporta histórico (`save`) e produz gráficos com `plot` para auditoria visual.
- **Padrões de design:** 
  - _Facade_ – `compute_stage_metrics` oculta detalhes de cada família de métricas.
  - _Observer/Memento_-like – `MetricsTracker` registra estados por época, permitindo inspeção posterior sem acoplar-se ao loop de treino.
- **Integrações:** Chamado por scripts de treino/validação para logging e pós-análise (ex.: `scripts/009_analyze_stage2_confusion.py`).

### `models.py`
- **Propósito:** Implementar arquiteturas baseadas em ResNet-18 com atenção e cabeças especializadas por estágio.
- **Principais componentes:**
  - Blocos reutilizáveis (`SEBlock`, `SpatialAttention`, `ImprovedBackbone`).
  - Cabeças por estágio (`Stage1BinaryHead`, `Stage2ThreeWayHead`, `Stage3RectHead`, `Stage3ABHead`).
  - Modelos completos (`Stage1Model`, `Stage2Model`, `Stage3RectModel`, `Stage3ABModel`) que compõem backbone + head.
- **Classes e uso:**
  - `SEBlock`: módulo de atenção de canal aplicado após cada bloco residual.
  - `SpatialAttention`: módulo de atenção espacial complementar ao SEBlock.
  - `ImprovedBackbone`: backbone ResNet-18 customizado para entrada de 1 canal; reutilizado por todos os modelos.
  - `Stage1BinaryHead`: perceptron com temperatura ajustável para calibrar logits binários.
  - `Stage2ThreeWayHead`: MLP de duas camadas com dropout para classificação SPLIT/RECT/AB.
  - `Stage3RectHead`: classificador binário (HORZ vs VERT) com largura moderada.
  - `Stage3ABHead`: classificador 4-way com dropout mais alto para generalizar classes raras.
  - `Stage1Model`, `Stage2Model`, `Stage3RectModel`, `Stage3ABModel`: combinam `ImprovedBackbone` com a respectiva head, mantendo interface `forward(x)` compatível com PyTorch.
- **Padrões de design:** 
  - _Composition_ para combinar backbone compartilhado com heads específicas.
  - _Strategy_ no uso de diferentes cabeças mantendo a mesma interface `forward`.
  - _Parameter Sharing_ via `ImprovedBackbone` garante consistência entre estágios.
- **Integrações:** Modelos são instanciados nas rotinas de treino; `ensemble.py` opera sobre `Stage3ABModel`; perdas e métricas são conectadas via PyTorch Lightning/loops customizados.

## Fluxo típico de uso
1. **Carregamento dos dados:** `load_block_records` → `build_hierarchical_dataset_v6`, escolhendo `stage` e `augmentation`.
2. **Treino do modelo:** instanciar `StageXModel`, selecionar loss com `get_loss_function`, preparar `DataLoader` (eventualmente com samplers balanceados).
3. **Avaliação:** usar `compute_stage_metrics` e `print_metrics_summary`; gerar gráficos com `plot_confusion_matrix`.
4. **Especialização AB:** treinar múltiplos `Stage3ABModel`, empacotar com `ABEnsemble` ou variantes.
5. **Inferência/TTA:** aplicar `TestTimeAugmentation` e agregar previsões; para AB usar `predict_with_uncertainty` para medir concordância.

## Padrões transversais e decisões de design
- **Hierarquia de estágios:** evita que classes raras do Stage 3 competem diretamente com Stage 1, reduzindo ruído.
- **Factories declarativas:** `get_augmentation`, `build_hierarchical_dataset_v6`, `create_ab_ensemble`, `get_loss_function` permitem configurar o pipeline via dicionários e reduzem ramificações condicionais espalhadas pelo código.
- **Separação de responsabilidades:** cada script mantém escopo único (dados, augmentations, modelos, ensemble, perdas, métricas), favorecendo testes unitários e reuso.
- **Extensibilidade:** novas cabeças ou augmentations requerem apenas adicionar classes que respeitem os contratos (protótipos `__call__` ou `forward`) e registrar na respectiva factory.

## Recomendações para extensão
- Ao adicionar um novo estágio ou cabeças, reutilize `ImprovedBackbone` e exponha nova factory no script correspondente.
- Para novas estratégias de balanceamento, expanda `data_hub.py` com funções que retornem índices ou samplers customizados, mantendo a interface de `BlockRecord`.
- Novas métricas devem ser agregadas em `metrics.py`, de preferência envolvendo `compute_stage_metrics` para preservar a façade única.
