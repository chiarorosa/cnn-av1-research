"""Pipeline v6 para Particionamento AV1 com Hierarquia CNN.

Este módulo implementa uma arquitetura hierárquica de deep learning para predição
de particionamento de blocos AV1 em codificação de vídeo.

Mudanças principais vs v5:
- Stage 2 simplificado (3 classes: SPLIT, RECT, AB)
- Ensemble de 3 modelos para Stage 3-AB
- Focal Loss em todos os estágios
- Heavy data augmentation para classes minoritárias
- Attention mechanisms (SE-Block + SAM)
"""

from __future__ import annotations

__version__ = "6.0.0-alpha"
__author__ = "Chiaro Rosa"

# Definições de particionamento AV1 (mantidas de v5)
PARTITION_ID_TO_NAME = {
    0: "PARTITION_NONE",
    1: "PARTITION_HORZ",
    2: "PARTITION_VERT",
    3: "PARTITION_SPLIT",
    4: "PARTITION_HORZ_A",
    5: "PARTITION_HORZ_B",
    6: "PARTITION_VERT_A",
    7: "PARTITION_VERT_B",
    8: "PARTITION_HORZ_4",
    9: "PARTITION_VERT_4",
}

PARTITION_NAME_TO_ID = {name: idx for idx, name in PARTITION_ID_TO_NAME.items()}

# Stage 1: Binary classification (mantido de v5)
# 0 = PARTITION_NONE, 1 = Particionado (qualquer outro tipo)

# Stage 2: Macro-classes (REDESENHADO em v6)
# Removemos NONE (já filtrado) e 1TO4 (não existe no dataset)
STAGE2_GROUPS = {
    "SPLIT": ["PARTITION_SPLIT"],
    "RECT": ["PARTITION_HORZ", "PARTITION_VERT"],
    "AB": [
        "PARTITION_HORZ_A",
        "PARTITION_HORZ_B",
        "PARTITION_VERT_A",
        "PARTITION_VERT_B",
    ],
}

STAGE2_NAME_TO_ID = {name: idx for idx, name in enumerate(STAGE2_GROUPS.keys())}
STAGE2_ID_TO_NAME = {idx: name for name, idx in STAGE2_NAME_TO_ID.items()}

# Stage 3: Specialists (mantido de v5, mas AB será ensemble)
STAGE3_GROUPS = {
    "RECT": ["PARTITION_HORZ", "PARTITION_VERT"],
    "AB": [
        "PARTITION_HORZ_A",
        "PARTITION_HORZ_B",
        "PARTITION_VERT_A",
        "PARTITION_VERT_B",
    ],
}

# Mapeamento de labels Stage 0 → Stage 1
def _label_to_stage1(label: int) -> int:
    """Converte label original (0-9) para Stage 1 (0=NONE, 1=PARTITION)."""
    return 0 if label == 0 else 1


# Mapeamento de labels Stage 0 → Stage 2
def _label_to_stage2(label: int) -> int:
    """Converte label original para Stage 2 (SPLIT=0, RECT=1, AB=2, NONE=-1)."""
    name = PARTITION_ID_TO_NAME[label]
    
    if name == "PARTITION_NONE":
        return -1  # Não deveria chegar aqui (filtrado no Stage 1)
    
    for group_name, members in STAGE2_GROUPS.items():
        if name in members:
            return STAGE2_NAME_TO_ID[group_name]
    
    return -1  # Desconhecido (ex: HORZ_4, VERT_4)


# Mapeamento de labels Stage 0 → Stage 3
def _label_to_stage3(label: int, head: str) -> int:
    """Converte label original para Stage 3 do especialista específico.
    
    Args:
        label: Label original (0-9)
        head: Nome do especialista ("RECT" ou "AB")
    
    Returns:
        Índice da classe no especialista, ou -1 se não pertence a este head
    """
    name = PARTITION_ID_TO_NAME[label]
    
    if head not in STAGE3_GROUPS:
        raise ValueError(f"Especialista desconhecido: {head}")
    
    group_members = STAGE3_GROUPS[head]
    
    if name in group_members:
        return group_members.index(name)
    
    return -1


# Exports principais
__all__ = [
    "__version__",
    "__author__",
    "PARTITION_ID_TO_NAME",
    "PARTITION_NAME_TO_ID",
    "STAGE2_GROUPS",
    "STAGE2_NAME_TO_ID",
    "STAGE2_ID_TO_NAME",
    "STAGE3_GROUPS",
]
