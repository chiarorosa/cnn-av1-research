#!/usr/bin/env python3
"""Pipeline inference utilities for the hierarchical AV1 CNN flow."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pesquisa.v5_pipeline import (  # noqa: E402
    PARTITION_ID_TO_NAME,
    PARTITION_NAME_TO_ID,
    STAGE2_GROUPS,
    STAGE2_NAME_TO_ID,
    STAGE3_GROUPS,
    build_hierarchical_model,
)


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------


class TensorDictDataset(Dataset):
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.tensors = tensors
        self.length = next(iter(tensors.values())).shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.tensors.items()}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_state_filtered(model: torch.nn.Module, checkpoint: Path, prefix: Optional[str] = None) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload.get("model_state", payload)
    if prefix is not None:
        state_dict = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
    else:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("specialist_heads.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] missing keys ao carregar {checkpoint.name}: {missing}")
    if unexpected:
        print(f"[WARN] unexpected keys ao carregar {checkpoint.name}: {unexpected}")


def build_pipeline_model(
    stage1_ckpt: Path,
    stage2_ckpt: Path,
    rect_ckpt: Optional[Path],
    ab_ckpt: Optional[Path],
    one2four_ckpt: Optional[Path],
    device: torch.device,
) -> torch.nn.Module:
    model = build_hierarchical_model(
        stage2_classes=len(STAGE2_GROUPS),
        specialist_classes={k: len(v) for k, v in STAGE3_GROUPS.items()},
        use_qp=False,
    )
    _load_state_filtered(model, stage1_ckpt)
    _load_state_filtered(model, stage2_ckpt)
    
    # Carregar especialistas apenas se os checkpoints foram fornecidos
    if rect_ckpt is not None:
        _load_state_filtered(model, rect_ckpt, prefix="specialist_heads.RECT")
    if ab_ckpt is not None:
        _load_state_filtered(model, ab_ckpt, prefix="specialist_heads.AB")
    if one2four_ckpt is not None:
        _load_state_filtered(model, one2four_ckpt, prefix="specialist_heads.1TO4")
    
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _compute_confusion(num_classes: int, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    return conf


def _macro_f1(conf: np.ndarray) -> float:
    tp = np.diag(conf)
    support = conf.sum(axis=1)
    predicted = conf.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return float(np.mean(f1))


@dataclass
class Stage1Metrics:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-9)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-9)

    @property
    def f1(self) -> float:
        prec = self.precision
        rec = self.recall
        return 2 * prec * rec / (prec + rec + 1e-9)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def run_pipeline(
    bundle: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    stage1_threshold: float = 0.5,
    batch_size: int = 256,
    csv_path: Optional[Path] = None,
    available_specialists: Optional[List[str]] = None,
) -> Dict[str, object]:
    dataset = TensorDictDataset(bundle)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    stage2_names = list(STAGE2_GROUPS.keys())
    stage3_heads = {name: list(labels) for name, labels in STAGE3_GROUPS.items()}
    
    # Se não foi especificado, assumir que todos estão disponíveis
    if available_specialists is None:
        available_specialists = list(STAGE3_GROUPS.keys())
    
    num_classes = len(PARTITION_ID_TO_NAME)

    final_conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    stage2_conf = np.zeros((len(stage2_names), len(stage2_names)), dtype=np.int64)
    stage3_conf: Dict[str, np.ndarray] = {
        head: np.zeros((len(labels), len(labels)), dtype=np.int64)
        for head, labels in stage3_heads.items()
    }

    stage1_metrics = Stage1Metrics(tp=0, fp=0, fn=0)
    stage1_total = 0
    stage1_correct = 0

    records: List[Dict[str, object]] = []

    with torch.no_grad():
        for idx_batch, batch_data in enumerate(loader):
            images = batch_data["image"].to(device)
            qp = batch_data.get("qp")
            if qp is not None:
                qp = qp.to(device)
            labels_stage0 = batch_data["label_stage0"].numpy()
            labels_stage1 = batch_data["label_stage1"].numpy()
            labels_stage2 = batch_data["label_stage2"].numpy()

            outputs = model(image=images, qp=qp)
            stage1_logits = outputs.stage1
            stage2_logits = outputs.stage2
            specialist_logits = outputs.specialists

            stage1_probs = torch.sigmoid(stage1_logits)
            stage1_pred = (stage1_probs >= stage1_threshold).long().cpu().numpy()

            stage1_total += len(labels_stage1)
            stage1_correct += int((stage1_pred == labels_stage1).sum())
            stage1_metrics.tp += int(((stage1_pred == 1) & (labels_stage1 == 1)).sum())
            stage1_metrics.fp += int(((stage1_pred == 1) & (labels_stage1 == 0)).sum())
            stage1_metrics.fn += int(((stage1_pred == 0) & (labels_stage1 == 1)).sum())

            stage2_pred = torch.argmax(stage2_logits, dim=1).cpu().numpy()
            partition_mask = labels_stage1 == 1
            if partition_mask.any():
                stage2_conf += _compute_confusion(len(stage2_names), labels_stage2[partition_mask], stage2_pred[partition_mask])

            for i in range(len(labels_stage0)):
                true_id = int(labels_stage0[i])
                if stage1_pred[i] == 0:
                    final_name = "PARTITION_NONE"
                    stage2_idx = None
                    stage3_name = None
                else:
                    stage2_idx = stage2_pred[i]
                    macro_name = stage2_names[stage2_idx]
                    if macro_name == "NONE":
                        final_name = "PARTITION_NONE"
                        stage3_name = None
                    elif macro_name == "SPLIT":
                        final_name = "PARTITION_SPLIT"
                        stage3_name = None
                    elif macro_name in stage3_heads and macro_name in available_specialists:
                        # Especialista disponível - usar predição do Stage3
                        head_logits = specialist_logits[macro_name][i]
                        head_pred = int(torch.argmax(head_logits).item())
                        final_name = stage3_heads[macro_name][head_pred]
                        head_true_tensor = batch_data[f"label_stage3_{macro_name}"]
                        head_true = int(head_true_tensor[i].item()) if head_true_tensor is not None else -1
                        stage3_name = macro_name
                        if head_true >= 0:
                            stage3_conf[macro_name][head_true, head_pred] += 1
                    elif macro_name in stage3_heads and macro_name not in available_specialists:
                        # Especialista NÃO disponível - fallback para primeira opção do grupo
                        # ou usar PARTITION_NONE como fallback seguro
                        final_name = stage3_heads[macro_name][0] if stage3_heads[macro_name] else "PARTITION_NONE"
                        stage3_name = f"{macro_name}_FALLBACK"
                    else:
                        final_name = "PARTITION_NONE"
                        stage3_name = None

                pred_id = PARTITION_NAME_TO_ID[final_name]
                final_conf[true_id, pred_id] += 1

                if csv_path is not None:
                    records.append({
                        "index": idx_batch * batch_size + i,
                        "stage0_true": PARTITION_ID_TO_NAME[true_id],
                        "stage1_true": int(labels_stage1[i]),
                        "stage1_pred": int(stage1_pred[i]),
                        "stage2_true": stage2_names[int(labels_stage2[i])] if partition_mask[i] else "NONE",
                        "stage2_pred": stage2_names[int(stage2_pred[i])] if stage2_idx is not None else "NONE",
                        "final_pred": final_name,
                        "stage3_head": stage3_name,
                    })

    final_accuracy = final_conf.diagonal().sum() / final_conf.sum()

    stage3_metrics = {}
    for head, conf in stage3_conf.items():
        total = conf.sum()
        if total > 0:
            stage3_metrics[head] = {
                "macro_f1": _macro_f1(conf),
                "support": int(total),
            }

    report = {
        "final_accuracy": final_accuracy,
        "stage1": {
            "accuracy": stage1_correct / stage1_total,
            "precision": stage1_metrics.precision,
            "recall": stage1_metrics.recall,
            "f1": stage1_metrics.f1,
            "tp": stage1_metrics.tp,
            "fp": stage1_metrics.fp,
            "fn": stage1_metrics.fn,
        },
        "stage2": {
            "macro_f1": _macro_f1(stage2_conf),
            "confusion_matrix": stage2_conf.tolist(),
            "class_names": stage2_names,
        },
        "stage3": {
            "metrics": stage3_metrics,
            "confusion_matrices": {head: conf.tolist() for head, conf in stage3_conf.items()},
        },
        "confusion_matrix": final_conf.tolist(),
        "class_names": [name for _, name in sorted(PARTITION_ID_TO_NAME.items())],
        "stage1_threshold": stage1_threshold,
    }

    if csv_path is not None and records:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia o pipeline hierárquico AV1")
    parser.add_argument("--dataset-root", type=Path, default=Path("pesquisa/v5_dataset"))
    parser.add_argument("--block-size", choices=["8", "16", "32", "64"], default="16")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stage1-threshold", type=float, default=0.5)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--rect-checkpoint", type=Path, required=False, default=None,
                       help="Checkpoint do especialista RECT (opcional)")
    parser.add_argument("--ab-checkpoint", type=Path, required=False, default=None,
                       help="Checkpoint do especialista AB (opcional)")
    parser.add_argument("--one2four-checkpoint", type=Path, required=False, default=None, 
                       help="Checkpoint do especialista 1TO4 (opcional)")
    parser.add_argument("--output-json", type=Path, default=Path("pesquisa/logs/v5_pipeline_eval.json"))
    parser.add_argument("--csv-path", type=Path, default=None, help="Salva predições detalhadas em CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_root / f"block_{args.block_size}" / f"{args.split}.pt"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset split não encontrado: {dataset_path}")

    bundle = torch.load(dataset_path)
    device = torch.device(args.device)

    # Determinar quais especialistas estão disponíveis
    available_specialists = []
    if args.rect_checkpoint is not None:
        available_specialists.append("RECT")
    if args.ab_checkpoint is not None:
        available_specialists.append("AB")
    if args.one2four_checkpoint is not None:
        available_specialists.append("1TO4")
    
    print(f"Especialistas disponíveis: {available_specialists}")

    model = build_pipeline_model(
        stage1_ckpt=args.stage1_checkpoint,
        stage2_ckpt=args.stage2_checkpoint,
        rect_ckpt=args.rect_checkpoint,
        ab_ckpt=args.ab_checkpoint,
        one2four_ckpt=args.one2four_checkpoint,
        device=device,
    )

    report = run_pipeline(
        bundle=bundle,
        model=model,
        device=device,
        stage1_threshold=args.stage1_threshold,
        batch_size=args.batch_size,
        csv_path=args.csv_path,
        available_specialists=available_specialists,
    )

    report["dataset"] = str(dataset_path)
    report["available_specialists"] = available_specialists
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
