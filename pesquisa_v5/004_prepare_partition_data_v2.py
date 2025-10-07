#!/usr/bin/env python3
"""
Script para preparar dados de partição AV1 para o rearrange_video.py (Versão 2)

Este script processa arquivos partition_frame_X.txt extraídos do encoder AV1
e gera arquivos Excel (.xlsx) no formato esperado pelo rearrange_video.py.

=== EXECUÇÃO POR TERMINAL ===

Sintaxe:
    python prepare_partition_data_v2.py <pasta_com_SEQUENCIAS>

Exemplo:
    python prepare_partition_data_v2.py /home/user/sequencias_videos/
    
    ou
    
    ./prepare_partition_data_v2.py /home/user/sequencias_videos/

Pré-requisitos:
    - Python 3.x instalado
    - Bibliotecas: pandas, openpyxl
    - Instalar dependências: pip install pandas openpyxl

=== ENTRADA ESPERADA NA PASTA ===
- Diretórios de sequências de vídeo, cada um contendo:
  - partition_frame_0.txt, partition_frame_1.txt, etc.
- O nome do diretório é usado como nome do vídeo

=== SAÍDA GERADA ===
- [nome_sequencia]-intra-0.xlsx, [nome_sequencia]-intra-1.xlsx, etc.
- Arquivos Excel criados na pasta raiz fornecida
- Cada arquivo contém planilhas para blocos: 64x64 ~ 32x32 ~ 16x16 ~ 8x8
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def parse_partition_line(line):
    """
    Parseia uma linha do arquivo de partição.
    
    Formato correto: order_hint frame_type block_size row col partition_mode qp
    Exemplo: 0 0 9 4 8 3 120
    """
    parts = line.strip().split()
    if len(parts) != 7:
        return None
    
    try:
        return {
            'order_hint': int(parts[0]),      # ID único do frame
            'frame_type': int(parts[1]),      # 0: intra, 1: inter
            'block_size': int(parts[2]),      # 3: 8x8, 6: 16x16, 9: 32x32, 12: 64x64
            'row': int(parts[3]),             # linha (1 unidade = 4 pixels)
            'col': int(parts[4]),             # coluna (1 unidade = 4 pixels)
            'partition_mode': int(parts[5]),  # modo de partição (0-9)
            'qp': int(parts[6])               # quantization parameter
        }
    except ValueError:
        return None

def block_size_to_pixels(block_size_index):
    """
    Converte índice de tamanho de bloco para pixels.
    Baseado no mapeamento do 001_analyze_partition_data.py
    """
    mapping = {
        3: 8,    # 8x8 pixels (BLOCK_8X8)
        6: 16,   # 16x16 pixels (BLOCK_16X16)
        9: 32,   # 32x32 pixels (BLOCK_32X32)
        12: 64   # 64x64 pixels (BLOCK_64X64)
    }
    return mapping.get(block_size_index)

def process_partition_file(partition_file):
    """Processa um arquivo de partição e retorna dados organizados por frame."""
    
    # Extrair número do frame do nome do arquivo
    basename = os.path.basename(partition_file)
    if not basename.startswith('partition_frame_') or not basename.endswith('.txt'):
        print(f"ERRO: Arquivo inválido: {basename} (esperado: partition_frame_X.txt)")
        return {}
    
    try:
        frame_number_from_file = int(basename[16:-4])  # partition_frame_X.txt -> X
    except ValueError:
        print(f"ERRO: Não foi possível extrair número do frame de: {basename}")
        return {}
    
    # Agora incluindo suporte para blocos 8x8
    frames_data = {frame_number_from_file: {64: [], 32: [], 16: [], 8: []}}
    order_hints_found = set()
    
    try:
        with open(partition_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                data = parse_partition_line(line)
                if data is None:
                    print(f"AVISO: Linha inválida em {basename}:{line_num}: {line.strip()}")
                    continue
                
                order_hints_found.add(data['order_hint'])
                
                # Validar se order_hint corresponde ao número do arquivo
                if data['order_hint'] != frame_number_from_file:
                    print(f"AVISO: {basename}: order_hint={data['order_hint']} não corresponde ao frame {frame_number_from_file}")
                
                block_pixels = block_size_to_pixels(data['block_size'])
                if block_pixels:
                    # Filtrar apenas blocos de frames intra (frame_type = 0)
                    if data['frame_type'] == 0:
                        frames_data[frame_number_from_file][block_pixels].append(data)
        
        # Verificar consistência dos order_hints
        if len(order_hints_found) > 1:
            print(f"AVISO: {basename}: múltiplos order_hints encontrados: {sorted(order_hints_found)}")
        elif len(order_hints_found) == 1 and frame_number_from_file in order_hints_found:
            print(f"OK: {basename}: frame {frame_number_from_file} validado (order_hint correto)")
        
        # Atualizado para incluir blocos 8x8
        total_blocks = sum(len(frames_data[frame_number_from_file][size]) for size in [64, 32, 16, 8])
        intra_blocks = sum(1 for size in [64, 32, 16, 8] for block in frames_data[frame_number_from_file][size] if block['frame_type'] == 0)
        print(f"  Blocos processados: {total_blocks} (intra: {intra_blocks})")
        
        # Mostrar distribuição por tamanho
        for size in [64, 32, 16, 8]:
            count = len(frames_data[frame_number_from_file][size])
            if count > 0:
                print(f"    {size}x{size}: {count} blocos")
                    
    except Exception as e:
        print(f"ERRO: Erro ao processar {partition_file}: {e}")
        return {}
    
    return frames_data

def create_excel_for_frame(frame_data, video_name, frame_number, output_dir, middle_word="data"):
    """Cria arquivo Excel para um frame específico."""
    excel_filename = f"{video_name}-{middle_word}-{frame_number}.xlsx"
    excel_path = os.path.join(output_dir, excel_filename)
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Criar planilhas para cada tamanho de bloco (incluindo 8x8)
            for block_size in [64, 32, 16, 8]:
                if block_size in frame_data and frame_data[block_size]:
                    # Criar DataFrame com os dados
                    df_data = []
                    for block in frame_data[block_size]:
                        df_data.append([
                            block['row'],            # Coluna A (linha do bloco)
                            block['col'],            # Coluna B (coluna do bloco) - usado pelo rearrange_video.py
                            block['partition_mode'], # Coluna C (modo de partição)
                            block['qp'],             # Coluna D (QP)
                            block['frame_type'],     # Coluna E (tipo do frame)
                            block['order_hint']      # Coluna F (order hint)
                        ])
                    
                    df = pd.DataFrame(df_data)
                    # Ordenar por linha e depois por coluna
                    df = df.sort_values([0, 1]).reset_index(drop=True)
                    
                    # Salvar na planilha SEM HEADER (header=False, index=False)
                    df.to_excel(writer, sheet_name=str(block_size), header=False, index=False)
                else:
                    # Criar planilha vazia se não houver dados
                    empty_df = pd.DataFrame()
                    empty_df.to_excel(writer, sheet_name=str(block_size), header=False, index=False)
        
        print(f"Criado: {excel_filename}")
        return True
        
    except Exception as e:
        print(f"Erro ao criar {excel_filename}: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Uso: python prepare_partition_data_v2.py <pasta_com_SEQUENCIAS>")
        print("\nA pasta deve conter diretórios de sequências de vídeo, cada um com:")
        print("  - partition_frame_X.txt (arquivos de partição)")
        print("  - O nome do diretório é usado como nome do vídeo")
        print("\nVERSÃO 2: Agora com suporte para blocos 8x8 (bsize=3)")
        sys.exit(1)
    
    sequences_dir = sys.argv[1]
    
    if not os.path.isdir(sequences_dir):
        print(f"Erro: '{sequences_dir}' não é um diretório válido.")
        sys.exit(1)
    
    # Buscar diretórios de sequências
    sequence_dirs = []
    for item in os.listdir(sequences_dir):
        item_path = os.path.join(sequences_dir, item)
        if os.path.isdir(item_path):
            # Verificar se o diretório contém arquivos partition_frame_*.txt
            partition_pattern = os.path.join(item_path, 'partition_frame_*.txt')
            partition_files = glob.glob(partition_pattern)
            if partition_files:
                sequence_dirs.append((item_path, item))  # (caminho_completo, nome_sequencia)
    
    if not sequence_dirs:
        print(f"Erro: Nenhum diretório com arquivos 'partition_frame_*.txt' encontrado em '{sequences_dir}'")
        sys.exit(1)
    
    print(f"Encontradas {len(sequence_dirs)} sequências de vídeo:")
    for _, sequence_name in sequence_dirs:
        print(f"  - {sequence_name}")
    
    print("\nVERSÃO 2: Processando blocos 64x64, 32x32, 16x16 e 8x8")
    
    # Processar cada sequência
    total_files_created = 0
    
    for sequence_path, sequence_name in sequence_dirs:
        print(f"\n=== Processando sequência: {sequence_name} ===")
        
        # Encontrar arquivos de partição na sequência
        partition_pattern = os.path.join(sequence_path, 'partition_frame_*.txt')
        partition_files = glob.glob(partition_pattern)
        
        print(f"Encontrados {len(partition_files)} arquivos de partição em {sequence_name}")
        
        # Processar todos os arquivos de partição da sequência
        all_frames_data = {}
        
        for partition_file in sorted(partition_files):
            print(f"Processando: {os.path.basename(partition_file)}")
            frames_data = process_partition_file(partition_file)
            
            # Mesclar dados de frames
            for frame_num, data in frames_data.items():
                if frame_num not in all_frames_data:
                    all_frames_data[frame_num] = {64: [], 32: [], 16: [], 8: []}
                
                # Atualizado para incluir blocos 8x8
                for block_size in [64, 32, 16, 8]:
                    all_frames_data[frame_num][block_size].extend(data[block_size])
        
        if not all_frames_data:
            print(f"AVISO: Nenhum dado válido encontrado para sequência {sequence_name}")
            continue
        
        print(f"\nResumo de frames encontrados para {sequence_name}: {sorted(all_frames_data.keys())}")
        for frame_num in sorted(all_frames_data.keys()):
            # Atualizado para incluir blocos 8x8
            total_blocks = sum(len(all_frames_data[frame_num][size]) for size in [64, 32, 16, 8])
            print(f"  Frame {frame_num}: {total_blocks} blocos intra")
            
            # Mostrar distribuição detalhada por tamanho
            for size in [64, 32, 16, 8]:
                count = len(all_frames_data[frame_num][size])
                if count > 0:
                    print(f"    {size}x{size}: {count} blocos")
        
        # Criar arquivos Excel para cada frame na pasta raiz
        print(f"\nCriando arquivos Excel para {len(all_frames_data)} frames de {sequence_name}...")
        
        created_count = 0
        middle_word = "intra"  # Filtra apenas blocos intra
        for frame_number in sorted(all_frames_data.keys()):
            if create_excel_for_frame(all_frames_data[frame_number], sequence_name, frame_number, sequences_dir, middle_word):
                created_count += 1
        
        print(f"Criados {created_count} arquivos Excel para sequência {sequence_name}")
        total_files_created += created_count
        
        print(f"Arquivos Excel de {sequence_name} salvos em: {sequences_dir}")
        for frame_number in sorted(all_frames_data.keys()):
            excel_name = f"{sequence_name}-{middle_word}-{frame_number}.xlsx"
            print(f"  {excel_name}")
    
    print(f"\n=== PROCESSO CONCLUÍDO ===")
    print(f"Total de arquivos Excel criados: {total_files_created}")
    print(f"Arquivos salvos em: {sequences_dir}")
    print(f"\nAgora você pode usar o rearrange_video.py com os arquivos Excel gerados.")

if __name__ == "__main__":
    main()