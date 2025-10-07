#!/usr/bin/env python3
"""
Script para consolidar amostras de treinamento de diferentes tamanhos de bloco.

Este script consolida múltiplos arquivos *_intra_raw_*.txt em um único arquivo
por tamanho de bloco, preparando os dados no formato adequado para CNN.

VERSÃO 2: Suporte para renomeação de arquivo único ou consolidação de múltiplos.

Uso: python 006_merge_sample_v2.py <pasta_intra_raw_blocks>

Exemplo: python 006_merge_sample_v2.py /home/experimentos/ai_ugc/intra_raw_blocks
"""

import glob
import os
import sys
import shutil

import numpy as np


def extract_sequence_name_from_raw_file(filename):
    """
    Extrai o nome da sequência de um arquivo intra_raw.
    
    Exemplo: '8_NewsClip_1080P-48ae-intra-0_intra_raw_64.txt' -> '8_NewsClip_1080P-48ae'
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Remover "_intra_raw_XX" do final
    parts = name_without_ext.split('_intra_raw_')
    if len(parts) == 2:
        # Remover "-intra-X" do final da primeira parte
        first_part = parts[0]
        intra_parts = first_part.split('-intra-')
        if len(intra_parts) >= 2:
            sequence_name = '-intra-'.join(intra_parts[:-1])
        else:
            sequence_name = first_part
    else:
        sequence_name = name_without_ext
    
    return sequence_name


def process_block_size(input_dir, image_size):
    """
    Processa arquivos de um tamanho de bloco específico.
    Retorna True se processou com sucesso, False caso contrário.
    """
    image_size_str = str(image_size)
    search_pattern = os.path.join(input_dir, f'*_intra_raw_{image_size_str}.txt')
    
    print(f"\n=== Processando blocos {image_size}x{image_size} ===")
    print(f"Padrão de busca: {search_pattern}")
    
    # Encontrar todos os arquivos que correspondem ao padrão
    matching_files = sorted(glob.glob(search_pattern))
    
    if not matching_files:
        print(f"⚠️ Nenhum arquivo encontrado para blocos {image_size}x{image_size}")
        return False
    
    print(f"Arquivos encontrados: {len(matching_files)}")
    for f in matching_files:
        print(f"  {os.path.basename(f)}")
    
    # Agrupar arquivos por sequência
    sequences = {}
    for filename in matching_files:
        sequence_name = extract_sequence_name_from_raw_file(filename)
        if sequence_name not in sequences:
            sequences[sequence_name] = []
        sequences[sequence_name].append(filename)
    
    print(f"Sequências encontradas: {len(sequences)}")
    for seq_name, files in sequences.items():
        print(f"  {seq_name}: {len(files)} arquivos")
    
    # Processar cada sequência
    for sequence_name, files in sequences.items():
        output_filename = f"{sequence_name}_sample_{image_size_str}.txt"
        output_path = os.path.join(input_dir, output_filename)
        
        if len(files) == 1:
            # Caso especial: apenas um arquivo, renomear
            source_file = files[0]
            print(f"\n📝 Renomeando arquivo único:")
            print(f"  De: {os.path.basename(source_file)}")
            print(f"  Para: {output_filename}")
            
            try:
                # ✅ CORREÇÃO: Usar move() ao invés de copy2() para realmente renomear
                shutil.move(source_file, output_path)
                print(f"  ✅ Arquivo renomeado com sucesso (arquivo original removido)")
                
                # Verificação rápida
                with open(output_path, 'rb') as f:
                    data = np.frombuffer(f.read(), dtype=np.uint8)
                    expected_pixels = image_size * image_size
                    num_blocks = len(data) // expected_pixels
                    print(f"  📊 Blocos no arquivo: {num_blocks}")
                    
            except Exception as e:
                print(f"  ❌ ERRO ao renomear: {e}")
                continue
                
        else:
            # Múltiplos arquivos: consolidar
            print(f"\n🔗 Consolidando {len(files)} arquivos para sequência {sequence_name}")
            
            all_data = []
            file_count = 0
            
            for filename in sorted(files):
                print(f"  Processando: {os.path.basename(filename)}")
                
                try:
                    with open(filename, 'rb') as f_samples:
                        file_data = f_samples.read()
                        training_data = np.frombuffer(file_data, dtype=np.uint8)
                        print(f"    Shape: {training_data.shape}")
                        
                        # Validação básica
                        expected_pixels_per_block = image_size * image_size
                        if len(training_data) % expected_pixels_per_block != 0:
                            print(f"    ⚠️ AVISO: Tamanho inconsistente")
                            print(f"    Pixels: {len(training_data)}, Esperado múltiplo de: {expected_pixels_per_block}")
                        
                        all_data.append(training_data)
                        file_count += 1
                        
                except Exception as e:
                    print(f"    ❌ ERRO: {e}")
                    continue
            
            # Consolidação
            if all_data:
                print(f"  Consolidando {file_count} arquivos...")
                
                consolidated_data = np.concatenate(all_data)
                print(f"  Dados consolidados shape: {consolidated_data.shape}")
                
                with open(output_path, 'wb') as f_output:
                    f_output.write(consolidated_data.tobytes())
                
                # Verificação
                with open(output_path, 'rb') as f_check:
                    check_data = f_check.read()
                    verification_data = np.frombuffer(check_data, dtype=np.uint8)
                    
                    expected_pixels_per_block = image_size * image_size
                    num_blocks = len(verification_data) // expected_pixels_per_block
                    
                    print(f"  ✅ Arquivo consolidado criado: {output_filename}")
                    print(f"  📊 Total de blocos: {num_blocks}")
                    print(f"  📊 Shape final: {verification_data.shape}")
            else:
                print(f"  ❌ Nenhum dado válido para consolidar")
    
    return True


def main():
    """Função principal do script."""
    # Validação dos argumentos
    if len(sys.argv) != 2:
        print("ERRO: Número incorreto de argumentos!")
        print("Uso: python 006_merge_sample_v2.py <pasta_intra_raw_blocks>")
        print("\nExemplo:")
        print("  python 006_merge_sample_v2.py /home/experimentos/ai_ugc/intra_raw_blocks")
        print("\nDescrição:")
        print("  pasta_intra_raw_blocks: Diretório contendo arquivos *_intra_raw_*.txt")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    
    # Validação do diretório
    if not os.path.isdir(input_dir):
        print(f"ERRO: Diretório não encontrado: {input_dir}")
        sys.exit(1)
    
    print(f"=== MERGE SAMPLE V2 - CONSOLIDAÇÃO E RENOMEAÇÃO ===")
    print(f"Diretório de entrada: {input_dir}")
    
    # Processamento para diferentes tamanhos de bloco
    processed_count = 0
    image_size = 64
    
    for i in range(4):  # Processa blocos: 64x64, 32x32, 16x16, 8x8
        success = process_block_size(input_dir, image_size)
        if success:
            processed_count += 1
        image_size = image_size // 2
    
    print(f"\n=== PROCESSAMENTO CONCLUÍDO ===")
    print(f"Tamanhos de bloco processados: {processed_count}/4")
    print(f"Arquivos de saída gerados em: {input_dir}")


if __name__ == "__main__":
    main()