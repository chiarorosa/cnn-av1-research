#!/usr/bin/env python3
"""
Script 009: Análise de Confusion Matrix do Stage 2

Objetivo:
    Computar a matriz de confusão real do Stage 2 (3-way: SPLIT, RECT, AB)
    para uso no Experimento 10 (Confusion-Based Noise Injection).

Fundamentação:
    - Heigold et al. (2016): Train-with-predictions reduz cascade error
    - Hendrycks et al. (2019): Noise realista > noise uniforme
    - Natarajan et al. (2013): Distribuição de noise importa para robustez

Hipótese H3.2:
    Usar distribuição de confusão real (não labels aleatórios) melhorará
    accuracy mantendo robustez.

Autor: Chiaro Rosa
Data: 14/10/2025
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

# Adicionar v6_pipeline ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'v6_pipeline'))

from models import Stage2Model


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description='Analisar confusion matrix do Stage 2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--stage2-model',
        type=str,
        required=True,
        help='Path para checkpoint do Stage 2 (frozen model, epoch 8)'
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='pesquisa_v6/v6_dataset/block_16',
        help='Path para diretório do dataset'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='pesquisa_v6/logs/v6_experiments/confusion_matrix_stage2.json',
        help='Path para salvar confusion matrix JSON'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device para inferência'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size para inferência'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Modo teste: usar apenas 1000 samples'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> nn.Module:
    """
    Carrega modelo Stage 2 do checkpoint.
    
    Args:
        checkpoint_path: Path para .pt file
        device: 'cuda' ou 'cpu'
    
    Returns:
        Modelo carregado em eval mode
    """
    print(f"🔄 Carregando modelo de: {checkpoint_path}")
    
    # Criar modelo
    model = Stage2Model(pretrained=False)
    
    # Carregar state_dict
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Checkpoint pode ter formato diferente
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modelo carregado com sucesso")
    return model


def compute_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computa predições do modelo no dataset.
    
    Args:
        model: Modelo Stage 2
        dataloader: DataLoader do validation set
        device: 'cuda' ou 'cpu'
    
    Returns:
        (predictions, ground_truths) como numpy arrays
    """
    predictions = []
    ground_truths = []
    
    print(f"🔄 Computando predições em {len(dataloader)} batches...")
    
    with torch.no_grad():
        for blocks, labels, qps in tqdm(dataloader, desc="Inferência"):
            blocks = blocks.to(device)
            
            # Forward pass
            outputs = model(blocks)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.append(preds.cpu().numpy())
            ground_truths.append(labels.numpy())
    
    # Concatenar
    predictions = np.concatenate(predictions)
    ground_truths = np.concatenate(ground_truths)
    
    print(f"✅ {len(predictions)} predições computadas")
    
    return predictions, ground_truths


def analyze_confusion_matrix(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    class_names: list[str]
) -> dict:
    """
    Analisa confusion matrix e extrai probabilidades de erro.
    
    Args:
        predictions: Array de predições
        ground_truths: Array de labels verdadeiros
        class_names: Lista de nomes das classes ['SPLIT', 'RECT', 'AB']
    
    Returns:
        Dicionário com confusion matrix e probabilidades
    """
    print("🔄 Computando confusion matrix...")
    
    # Computar confusion matrix
    cm = confusion_matrix(ground_truths, predictions)
    
    print("\n📊 Confusion Matrix (absoluta):")
    print("     Pred: SPLIT    RECT      AB")
    for i, class_name in enumerate(class_names):
        print(f"GT {class_name:5s}: {cm[i, 0]:6d}  {cm[i, 1]:6d}  {cm[i, 2]:6d}")
    
    # Normalizar por linha (cada GT class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print("\n📊 Confusion Matrix (normalizada - probabilidades):")
    print("     Pred:  SPLIT    RECT      AB")
    for i, class_name in enumerate(class_names):
        print(f"GT {class_name:5s}: {cm_normalized[i, 0]:6.3f}  {cm_normalized[i, 1]:6.3f}  {cm_normalized[i, 2]:6.3f}")
    
    # Extrair probabilidades de confusão para cada classe
    confusion_probs = {}
    
    for i, gt_class in enumerate(class_names):
        confusion_probs[gt_class] = {}
        
        for j, pred_class in enumerate(class_names):
            if i == j:
                # Acurácia (diagonal)
                confusion_probs[gt_class]['correct'] = float(cm_normalized[i, j])
            else:
                # Confusão (fora da diagonal)
                confusion_probs[gt_class][pred_class] = float(cm_normalized[i, j])
    
    # Classification report
    print("\n📊 Classification Report:")
    unique_labels = sorted(list(set(predictions.tolist() + ground_truths.tolist())))
    print(f"Classes encontradas: {unique_labels}")
    print(classification_report(
        ground_truths,
        predictions,
        labels=unique_labels,
        target_names=[class_names[i] if i < len(class_names) else f"CLASS_{i}" for i in unique_labels],
        digits=4,
        zero_division=0
    ))
    
    # Estatísticas adicionais
    accuracy = np.mean(predictions == ground_truths)
    print(f"\n📊 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Análise de erros mais comuns
    print("\n🔍 Principais Confusões (Top 3):")
    errors = []
    for i, gt_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            if i != j:
                count = cm[i, j]
                prob = cm_normalized[i, j]
                errors.append((gt_class, pred_class, count, prob))
    
    errors.sort(key=lambda x: x[2], reverse=True)
    for gt_class, pred_class, count, prob in errors[:3]:
        print(f"  GT={gt_class:5s} → Pred={pred_class:5s}: {count:6d} samples ({prob*100:.2f}%)")
    
    return {
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'confusion_probabilities': confusion_probs,
        'accuracy': float(accuracy),
        'class_names': class_names,
        'total_samples': int(len(predictions))
    }


def save_confusion_data(confusion_data: dict, output_path: str):
    """
    Salva dados de confusão em JSON.
    
    Args:
        confusion_data: Dicionário com confusion matrix e probabilidades
        output_path: Path para salvar JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Salvando confusion data em: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(confusion_data, f, indent=2)
    
    print(f"✅ Confusion data salva com sucesso")
    
    # Também salvar versão simplificada (só probabilidades)
    simple_path = output_path.parent / f"{output_path.stem}_simple.json"
    simple_data = {
        'confusion_probabilities': confusion_data['confusion_probabilities'],
        'accuracy': confusion_data['accuracy'],
        'class_names': confusion_data['class_names']
    }
    
    with open(simple_path, 'w') as f:
        json.dump(simple_data, f, indent=2)
    
    print(f"💾 Versão simplificada salva em: {simple_path}")


def main():
    """Função principal."""
    args = parse_args()
    
    print("=" * 80)
    print("Script 009: Análise de Confusion Matrix do Stage 2")
    print("=" * 80)
    print(f"Stage 2 model: {args.stage2_model}")
    print(f"Dataset dir:   {args.dataset_dir}")
    print(f"Output:        {args.output}")
    print(f"Device:        {args.device}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Test mode:     {args.test}")
    print("=" * 80)
    
    # 1. Carregar dataset
    print("\n📂 Carregando validation dataset...")
    dataset_dir = Path(args.dataset_dir)
    val_data = torch.load(dataset_dir / "val.pt", weights_only=False)
    
    # Extrair labels Stage 2
    labels_stage2 = val_data['labels_stage2']
    
    # CRÍTICO: Filtrar amostras que pertencem ao Stage 2 (remover NONE com label=-1)
    valid_mask = labels_stage2 != -1
    samples_filtered = val_data['samples'][valid_mask]
    labels_filtered = labels_stage2[valid_mask]
    
    print(f"📊 Estatísticas do dataset:")
    print(f"  Total samples: {len(labels_stage2)}")
    print(f"  Stage 2 samples (SPLIT/RECT/AB): {valid_mask.sum().item()}")
    print(f"  NONE samples (filtrados): {(~valid_mask).sum().item()}")
    
    # Verificar distribuição de classes
    unique, counts = torch.unique(labels_filtered, return_counts=True)
    print(f"  Distribuição Stage 2:")
    class_names_full = ['SPLIT', 'RECT', 'AB']
    for label, count in zip(unique.tolist(), counts.tolist()):
        class_name = class_names_full[label] if label < len(class_names_full) else f"CLASS_{label}"
        print(f"    {class_name}: {count} ({count/len(labels_filtered)*100:.1f}%)")
    
    # Criar dataset simples (só precisamos de blocks e labels)
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, samples, labels):
            self.samples = samples
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.samples[idx], self.labels[idx], 0  # qp dummy
    
    dataset = SimpleDataset(samples_filtered, labels_filtered)
    
    # Modo teste: usar apenas subset
    if args.test:
        print(f"⚠️  MODO TESTE: Usando apenas 1000 samples")
        indices = torch.randperm(len(dataset))[:1000].tolist()
        dataset.samples = dataset.samples[indices]
        dataset.labels = dataset.labels[indices]
    
    print(f"✅ Dataset carregado: {len(dataset)} samples")
    
    # 2. Criar dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # 3. Carregar modelo
    model = load_model(args.stage2_model, args.device)
    
    # 4. Computar predições
    predictions, ground_truths = compute_predictions(model, dataloader, args.device)
    
    # 5. Analisar confusion matrix
    class_names = ['SPLIT', 'RECT', 'AB']
    confusion_data = analyze_confusion_matrix(predictions, ground_truths, class_names)
    
    # 6. Salvar resultados
    save_confusion_data(confusion_data, args.output)
    
    print("\n" + "=" * 80)
    print("✅ Análise completa!")
    print("=" * 80)
    print(f"\n📄 Arquivos gerados:")
    print(f"  1. {args.output}")
    print(f"  2. {Path(args.output).parent / f'{Path(args.output).stem}_simple.json'}")
    print(f"\n🎯 Próximo passo:")
    print(f"  Usar 'confusion_probabilities' para treinar Stage 3 com noise realista")
    print(f"  (Experimento 10: Confusion-Based Noise Injection)")


if __name__ == '__main__':
    main()
