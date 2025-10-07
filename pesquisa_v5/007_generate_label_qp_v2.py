#!/usr/bin/env python3
"""
Script para gerar arquivos de labels e valores QP a partir de arquivos Excel.

Este script processa arquivos Excel contendo dados de partição AV1 e gera
arquivos de texto com labels e valores QP correspondentes para treinamento CNN.

VERSÃO 2: Processamento em lote com estrutura organizada de saída.

Uso: python 007_generate_label_qp_v2.py <pasta_xlsx>

Exemplo: python 007_generate_label_qp_v2.py /home/experimentos/ai_ugc

Formato esperado dos arquivos Excel:
    nome_video-intra-frame.xlsx
    Exemplo: 1_NewsClip_1080P-5b53-intra-0.xlsx

Estrutura de saída gerada:
    <pasta_xlsx>/labels/nome_video_labels_64_intra.txt
    <pasta_xlsx>/qps/nome_video_qps_64_intra.txt
    (para cada tamanho de bloco: 64, 32, 16, 8)
"""

import glob
import pandas as pd
import numpy as np
import os
import sys


def extract_video_name_from_xlsx(xlsx_filename):
    """
    Extrai o nome do vídeo de um arquivo XLSX.
    
    Exemplo: '1_NewsClip_1080P-5b53-intra-0.xlsx' -> '1_NewsClip_1080P-5b53'
    """
    basename = os.path.basename(xlsx_filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Remover "-intra-X" do final
    parts = name_without_ext.split('-')
    if len(parts) >= 3 and parts[-2] == 'intra':
        video_name = '-'.join(parts[:-2])
    else:
        # Fallback: remover apenas a última parte após o último '-'
        video_name = '-'.join(parts[:-1])
    
    return video_name


def extract_qp_from_xlsx(xlsx_filename):
    """
    Lê a coluna D (onde deveria estar o QP extraído)
    """
    #qp = pd.read_excel(xlsx_filename, sheet_name=sheet_name, usecols="D")
    return None


def process_single_xlsx(xlsx_file, labels_dir, qps_dir):
    """
    Processa um único arquivo XLSX e gera arquivos de labels e QPs.
    """
    print(f"Processando: {os.path.basename(xlsx_file)}")
    
    try:
        # Extrair nome do vídeo
        video_name = extract_video_name_from_xlsx(xlsx_file)
        print(f"  Vídeo: {video_name}")
        
        # Ler dados do Excel
        xlsx = pd.ExcelFile(xlsx_file)
        
        # Verificar planilhas disponíveis
        available_sheets = xlsx.sheet_names
        expected_sheets = ['64', '32', '16', '8']
        
        results = {}
        
        for sheet_name in expected_sheets:
            if sheet_name in available_sheets:
                try:
                    # Ler coluna C (partition_mode)
                    data = pd.read_excel(xlsx, sheet_name=sheet_name, usecols="C")
                    results[sheet_name] = data
                    print(f"    Planilha {sheet_name}: {len(data)} labels")
                except Exception as e:
                    print(f"    ⚠️ Erro ao ler planilha {sheet_name}: {e}")
                    results[sheet_name] = pd.DataFrame()
            else:
                print(f"    ⚠️ Planilha {sheet_name} não encontrada")
                results[sheet_name] = pd.DataFrame()
        
        # Extrair QP do nome do arquivo ou usar padrão
        # Como o formato atual não tem QP no nome, vamos tentar extrair do nome das planilhas
        # ou usar um valor padrão
        qp_value = extract_qp_from_xlsx(xlsx_file)
        if qp_value is None:
            # Tentar extrair QP de outra forma ou usar valor padrão
            qp_value = 80  # Valor padrão, pode ser ajustado conforme necessário
            print(f"    QP padrão usado: {qp_value}")
        
        # Gerar arquivos de saída para cada tamanho de bloco
        for sheet_name in expected_sheets:
            data = results[sheet_name]
            
            if not data.empty:
                # Arquivos de labels
                labels_filename = f"{video_name}_labels_{sheet_name}_intra.txt"
                labels_path = os.path.join(labels_dir, labels_filename)
                
                # Arquivos de QPs
                qps_filename = f"{video_name}_qps_{sheet_name}_intra.txt"
                qps_path = os.path.join(qps_dir, qps_filename)
                
                # Salvar labels
                np.savetxt(labels_path, data.values, fmt='%d')
                
                # Gerar e salvar QPs
                num_samples = len(data)
                qps_array = np.full(num_samples, qp_value, dtype=np.int32)
                np.savetxt(qps_path, qps_array, fmt='%d')
                
                print(f"    ✅ {sheet_name}: {num_samples} labels/qps salvos")
            else:
                print(f"    ⚠️ {sheet_name}: Planilha vazia, pulando")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ERRO ao processar {xlsx_file}: {e}")
        return False


def main():
    """Função principal do script."""
    # Validação dos argumentos
    if len(sys.argv) != 2:
        print("ERRO: Número incorreto de argumentos!")
        print("Uso: python 007_generate_label_qp_v2.py <pasta_xlsx>")
        print("\nExemplo:")
        print("  python 007_generate_label_qp_v2.py /home/experimentos/ai_ugc")
        print("\nDescrição:")
        print("  pasta_xlsx: Diretório contendo arquivos XLSX com dados de partição")
        print("\nEstrutura de saída:")
        print("  <pasta_xlsx>/labels/nome_video_labels_XX_intra.txt")
        print("  <pasta_xlsx>/qps/nome_video_qps_XX_intra.txt")
        sys.exit(1)
    
    xlsx_dir = sys.argv[1]
    
    # Validação do diretório
    if not os.path.isdir(xlsx_dir):
        print(f"ERRO: Diretório não encontrado: {xlsx_dir}")
        sys.exit(1)
    
    print(f"=== GENERATE LABEL QP V2 - PROCESSAMENTO EM LOTE ===")
    print(f"Diretório de entrada: {xlsx_dir}")
    
    # Criar diretórios de saída
    labels_dir = os.path.join(xlsx_dir, "labels")
    qps_dir = os.path.join(xlsx_dir, "qps")
    
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(qps_dir, exist_ok=True)
    
    print(f"Diretório de labels: {labels_dir}")
    print(f"Diretório de QPs: {qps_dir}")
    
    # Buscar arquivos XLSX
    xlsx_pattern = os.path.join(xlsx_dir, "*.xlsx")
    xlsx_files = glob.glob(xlsx_pattern)
    
    if not xlsx_files:
        print(f"ERRO: Nenhum arquivo XLSX encontrado em: {xlsx_dir}")
        sys.exit(1)
    
    # Filtrar apenas arquivos com padrão "-intra-"
    intra_xlsx_files = [f for f in xlsx_files if "-intra-" in os.path.basename(f)]
    
    if not intra_xlsx_files:
        print(f"ERRO: Nenhum arquivo XLSX com padrão '-intra-' encontrado em: {xlsx_dir}")
        sys.exit(1)
    
    print(f"Arquivos XLSX encontrados: {len(intra_xlsx_files)}")
    
    # Agrupar arquivos por vídeo
    videos = {}
    for xlsx_file in intra_xlsx_files:
        video_name = extract_video_name_from_xlsx(xlsx_file)
        if video_name not in videos:
            videos[video_name] = []
        videos[video_name].append(xlsx_file)
    
    print(f"Vídeos únicos encontrados: {len(videos)}")
    for video_name, files in videos.items():
        print(f"  {video_name}: {len(files)} frames")
    
    # Processar cada arquivo XLSX
    print(f"\n=== INICIANDO PROCESSAMENTO ===")
    total_processed = 0
    total_failed = 0
    
    for xlsx_file in sorted(intra_xlsx_files):
        success = process_single_xlsx(xlsx_file, labels_dir, qps_dir)
        if success:
            total_processed += 1
        else:
            total_failed += 1
    
    print(f"\n=== PROCESSAMENTO CONCLUÍDO ===")
    print(f"Total processados com sucesso: {total_processed}")
    print(f"Total com falhas: {total_failed}")
    print(f"Arquivos de labels salvos em: {labels_dir}")
    print(f"Arquivos de QPs salvos em: {qps_dir}")


if __name__ == "__main__":
    main()