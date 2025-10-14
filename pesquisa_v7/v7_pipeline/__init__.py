"""
V7 Pipeline - Clean Architecture for Thesis Solutions

Modules:
- backbone: ImprovedBackbone with SE-Block + Spatial Attention
- conv_adapter: Parameter-efficient adapters (Chen et al., CVPR 2024)
- ensemble: Multi-model voting (Ahad et al., 2024)
- hybrid_model: Adapter + Ensemble combination (novel)
- evaluation: Unified metrics and comparison
- data_hub: Dataset classes (from v6)
- losses: Loss functions (from v6)
"""

from .backbone import (
    ImprovedBackbone,
    ClassificationHead,
    create_stage1_head,
    create_stage2_head,
    create_stage3_rect_head,
    create_stage3_ab_head
)

from .conv_adapter import (
    ConvAdapter,
    AdapterBackbone,
    create_adapter_model
)

from .ensemble import (
    SoftVotingEnsemble,
    HardVotingEnsemble,
    HierarchicalEnsemble,
    DiverseBackboneEnsemble,
    create_ensemble_from_checkpoints,
    analyze_ensemble_diversity
)

from .hybrid_model import (
    AdapterEnsembleModel,
    HybridHierarchicalPipeline,
    create_hybrid_pipeline
)

from .evaluation import (
    MetricsCalculator,
    PipelineEvaluator,
    save_evaluation_results,
    load_evaluation_results,
    print_evaluation_report,
    PARTITION_NAMES
)

__all__ = [
    # Backbone
    'ImprovedBackbone',
    'ClassificationHead',
    'create_stage1_head',
    'create_stage2_head',
    'create_stage3_rect_head',
    'create_stage3_ab_head',
    
    # Conv-Adapter
    'ConvAdapter',
    'AdapterBackbone',
    'create_adapter_model',
    
    # Ensemble
    'SoftVotingEnsemble',
    'HardVotingEnsemble',
    'HierarchicalEnsemble',
    'DiverseBackboneEnsemble',
    'create_ensemble_from_checkpoints',
    'analyze_ensemble_diversity',
    
    # Hybrid
    'AdapterEnsembleModel',
    'HybridHierarchicalPipeline',
    'create_hybrid_pipeline',
    
    # Evaluation
    'MetricsCalculator',
    'PipelineEvaluator',
    'save_evaluation_results',
    'load_evaluation_results',
    'print_evaluation_report',
    'PARTITION_NAMES'
]

__version__ = '7.0.0'
