# Guia de Análise de Resultados - Experimento Adapter Capacity

**Experimento:** Aumento de capacidade do adapter (γ=4 → γ=2)  
**Script:** 020_train_adapter_solution.py  
**Output:** pesquisa_v7/logs/v7_experiments/solution1_adapter_reduction2/

---

## 1. Verificar Conclusão do Treinamento

### Comando para verificar status:
```bash
cd /home/chiarorosa/CNN_AV1
ls -lh pesquisa_v7/logs/v7_experiments/solution1_adapter_reduction2/stage2_adapter/
```

**Arquivos esperados**:
- ✅ `stage2_adapter_model_best.pt` - Melhor checkpoint
- ✅ `stage2_adapter_history.pt` - Histórico de treino
- ✅ `stage2_adapter_metrics.json` - Métricas finais
- ✅ `stage2_adapter_model_final.pt` - Checkpoint final (opcional)

**Se arquivos não existirem**: Treinamento ainda em andamento. Aguarde.

---

## 2. Extrair Métricas Principais

### Comando para ler métricas:
```bash
cd /home/chiarorosa/CNN_AV1
python3 -c "
import json
with open('pesquisa_v7/logs/v7_experiments/solution1_adapter_reduction2/stage2_adapter/stage2_adapter_metrics.json', 'r') as f:
    m = json.load(f)
    
print('=== STAGE 2 - ADAPTER REDUCTION=2 ===')
print(f'Val F1: {m[\"val_f1\"]*100:.2f}%')
print(f'Train F1: {m[\"train_f1\"]*100:.2f}%')
print(f'Train-Val Gap: {(m[\"train_f1\"] - m[\"val_f1\"])*100:.2f}%')
print(f'Best epoch: {m[\"best_epoch\"]}')
print(f'Total epochs: {m[\"total_epochs\"]}')
print()
print('Per-class (validation):')
for cls, metrics in m['val_per_class'].items():
    print(f'  {cls}: F1={metrics[\"f1\"]:.4f}, P={metrics[\"precision\"]:.4f}, R={metrics[\"recall\"]:.4f}')
"
```

### Comparação com Baseline (γ=4):
```bash
cd /home/chiarorosa/CNN_AV1
python3 -c "
import json

# Baseline (γ=4)
with open('pesquisa_v7/logs/v7_experiments/solution1_adapter/stage2_adapter/stage2_adapter_metrics.json', 'r') as f:
    m4 = json.load(f)

# Novo (γ=2)
with open('pesquisa_v7/logs/v7_experiments/solution1_adapter_reduction2/stage2_adapter/stage2_adapter_metrics.json', 'r') as f:
    m2 = json.load(f)

print('=== COMPARISON: γ=4 vs γ=2 ===')
print(f'Baseline (γ=4): Val F1 = {m4[\"val_f1\"]*100:.2f}%')
print(f'Experiment (γ=2): Val F1 = {m2[\"val_f1\"]*100:.2f}%')
print(f'Delta: {(m2[\"val_f1\"] - m4[\"val_f1\"])*100:.2f} percentage points')
print()
print(f'Baseline gap: {(m4[\"train_f1\"] - m4[\"val_f1\"])*100:.2f}%')
print(f'Experiment gap: {(m2[\"train_f1\"] - m2[\"val_f1\"])*100:.2f}%')
print()
print(f'Baseline epochs: {m4[\"total_epochs\"]} (best at {m4[\"best_epoch\"]})')
print(f'Experiment epochs: {m2[\"total_epochs\"]} (best at {m2[\"best_epoch\"]})')
"
```

---

## 3. Análise Visual das Curvas

### Plotar curvas de aprendizado:
```bash
cd /home/chiarorosa/CNN_AV1
python3 << 'EOF'
import torch
import matplotlib.pyplot as plt
import numpy as np

# Carregar histories
hist_g4 = torch.load('pesquisa_v7/logs/v7_experiments/solution1_adapter/stage2_adapter/stage2_adapter_history.pt', weights_only=False)
hist_g2 = torch.load('pesquisa_v7/logs/v7_experiments/solution1_adapter_reduction2/stage2_adapter/stage2_adapter_history.pt', weights_only=False)

# Criar figura
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Loss curves
axes[0, 0].plot(hist_g4['train_loss'], label='Train (γ=4)', linestyle='--', alpha=0.7)
axes[0, 0].plot(hist_g4['val_loss'], label='Val (γ=4)', linestyle='--', linewidth=2)
axes[0, 0].plot(hist_g2['train_loss'], label='Train (γ=2)', linestyle='-', alpha=0.7)
axes[0, 0].plot(hist_g2['val_loss'], label='Val (γ=2)', linestyle='-', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('Loss Curves: Adapter Capacity Comparison')

# 2. F1 curves
axes[0, 1].plot(hist_g4['train_f1'], label='Train (γ=4)', linestyle='--', alpha=0.7)
axes[0, 1].plot(hist_g4['val_f1'], label='Val (γ=4)', linestyle='--', linewidth=2)
axes[0, 1].plot(hist_g2['train_f1'], label='Train (γ=2)', linestyle='-', alpha=0.7)
axes[0, 1].plot(hist_g2['val_f1'], label='Val (γ=2)', linestyle='-', linewidth=2)
axes[0, 1].axhline(y=0.60, color='green', linestyle=':', label='Target (60%)', alpha=0.5)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('F1-score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('F1 Curves: Adapter Capacity Comparison')

# 3. Train-Val Gap
gap_g4 = np.array(hist_g4['train_f1']) - np.array(hist_g4['val_f1'])
gap_g2 = np.array(hist_g2['train_f1']) - np.array(hist_g2['val_f1'])
axes[1, 0].plot(gap_g4, label='γ=4 (166k params)', linestyle='--', linewidth=2)
axes[1, 0].plot(gap_g2, label='γ=2 (332k params)', linestyle='-', linewidth=2)
axes[1, 0].axhline(y=0.05, color='green', linestyle=':', label='Healthy (5%)', alpha=0.5)
axes[1, 0].axhline(y=0.08, color='orange', linestyle=':', label='Warning (8%)', alpha=0.5)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Train - Val F1')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('Generalization Gap')

# 4. Learning Rate Schedule
axes[1, 1].plot(hist_g4['learning_rates'], label='γ=4', linestyle='--', linewidth=2)
axes[1, 1].plot(hist_g2['learning_rates'], label='γ=2', linestyle='-', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_yscale('log')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('Learning Rate Schedule')

plt.tight_layout()
plt.savefig('pesquisa_v7/docs_v7/figures/adapter_capacity_comparison.png', dpi=150, bbox_inches='tight')
print('✅ Figura salva em: pesquisa_v7/docs_v7/figures/adapter_capacity_comparison.png')
plt.close()

# Figura adicional: Per-class F1 comparison
import json
with open('pesquisa_v7/logs/v7_experiments/solution1_adapter/stage2_adapter/stage2_adapter_metrics.json', 'r') as f:
    m4 = json.load(f)
with open('pesquisa_v7/logs/v7_experiments/solution1_adapter_reduction2/stage2_adapter/stage2_adapter_metrics.json', 'r') as f:
    m2 = json.load(f)

classes = list(m4['val_per_class'].keys())
f1_g4 = [m4['val_per_class'][c]['f1'] for c in classes]
f1_g2 = [m2['val_per_class'][c]['f1'] for c in classes]

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, f1_g4, width, label='γ=4', alpha=0.8)
bars2 = ax.bar(x + width/2, f1_g2, width, label='γ=2', alpha=0.8)

ax.set_xlabel('Class')
ax.set_ylabel('F1-score')
ax.set_title('Per-Class F1: Adapter Capacity Comparison')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Adicionar valores nas barras
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('pesquisa_v7/docs_v7/figures/adapter_capacity_per_class.png', dpi=150, bbox_inches='tight')
print('✅ Figura salva em: pesquisa_v7/docs_v7/figures/adapter_capacity_per_class.png')
EOF
```

**Nota:** Crie o diretório de figuras primeiro:
```bash
mkdir -p pesquisa_v7/docs_v7/figures
```

---

## 4. Inspeção do Checkpoint

### Verificar parâmetros do adapter:
```bash
cd /home/chiarorosa/CNN_AV1
python3 << 'EOF'
import torch

ckpt = torch.load('pesquisa_v7/logs/v7_experiments/solution1_adapter_reduction2/stage2_adapter/stage2_adapter_model_best.pt', weights_only=False)

# Encontrar parâmetros do adapter
model_keys = list(ckpt['model_state_dict'].keys())
adapter_keys = [k for k in model_keys if any(x in k for x in ['down_proj', 'up_proj', 'alpha', 'dw_conv'])]

print('=== ADAPTER ARCHITECTURE (γ=2) ===')
print(f'Total model keys: {len(model_keys)}')
print(f'Adapter-specific keys: {len(adapter_keys)}')
print()

# Contar parâmetros por layer
layer3_params = sum(ckpt['model_state_dict'][k].numel() for k in adapter_keys if 'layer3' in k)
layer4_params = sum(ckpt['model_state_dict'][k].numel() for k in adapter_keys if 'layer4' in k)

print(f'Layer 3 adapter params: {layer3_params:,}')
print(f'Layer 4 adapter params: {layer4_params:,}')
print(f'Total adapter params: {layer3_params + layer4_params:,}')
print()

# Mostrar shapes
print('=== PARAMETER SHAPES ===')
for k in sorted(adapter_keys):
    shape = ckpt['model_state_dict'][k].shape
    params = ckpt['model_state_dict'][k].numel()
    print(f'{k}: {shape} ({params:,} params)')

print()
print(f'Parameter efficiency: {ckpt["param_efficiency"]:.2f}%')
print(f'Best epoch: {ckpt["best_epoch"]}')
print(f'Val F1 at best: {ckpt["val_f1_at_best"]:.4f}')
EOF
```

---

## 5. Decisão: Adotar ou Não?

### Critérios de Decisão:

**✅ ADOTAR γ=2 se:**
1. ΔF1 ≥ 2% (ganho significativo, Chen et al. prediction)
2. Train-val gap 5-8% (saudável, sem overfitting)
3. Convergência estável (sem divergência)
4. Pelo menos 1 classe melhorou substancialmente

**⚠️ CONSIDERAR se:**
1. ΔF1 entre 1-2% (ganho marginal)
2. Train-val gap < 10% (ainda controlado)
3. Convergência mais lenta mas estável
4. Trade-off F1 vs params é aceitável (0.5-1.0 F1 points per 100k params)

**❌ REJEITAR γ=2 se:**
1. ΔF1 < 1% (sem ganho prático)
2. Train-val gap > 10% (overfitting severo)
3. Instabilidade no treino (loss oscilante)
4. Todas as classes pioraram ou mantiveram

### Comando para decisão automatizada:
```bash
cd /home/chiarorosa/CNN_AV1
python3 << 'EOF'
import json

# Carregar métricas
with open('pesquisa_v7/logs/v7_experiments/solution1_adapter/stage2_adapter/stage2_adapter_metrics.json', 'r') as f:
    m4 = json.load(f)
with open('pesquisa_v7/logs/v7_experiments/solution1_adapter_reduction2/stage2_adapter/stage2_adapter_metrics.json', 'r') as f:
    m2 = json.load(f)

# Calcular métricas de decisão
delta_f1 = (m2['val_f1'] - m4['val_f1']) * 100
gap_g2 = (m2['train_f1'] - m2['val_f1']) * 100

print('=== DECISION CRITERIA ===')
print(f'ΔF1: {delta_f1:+.2f} percentage points')
print(f'Train-Val Gap (γ=2): {gap_g2:.2f}%')
print()

# Decisão
if delta_f1 >= 2.0 and gap_g2 <= 8.0:
    print('✅ RECOMENDAÇÃO: ADOTAR γ=2')
    print('   Ganho significativo sem overfitting')
elif delta_f1 >= 1.0 and gap_g2 <= 10.0:
    print('⚠️  RECOMENDAÇÃO: CONSIDERAR γ=2')
    print('   Ganho marginal, avaliar trade-off')
else:
    print('❌ RECOMENDAÇÃO: MANTER γ=4')
    print('   Ganho insuficiente ou overfitting')
EOF
```

---

## 6. Atualizar Documentação

### Após decidir, atualizar:

**Arquivo:** `pesquisa_v7/docs_v7/02_experimento_adapter_capacity.md`

**Seção 4 - Resultados**: Preencher tabela com valores reais

**Seção 5 - Análise Comparativa**: Completar análise com conclusões

**Seção 6 - Curvas**: Adicionar figuras geradas

**Seção 7 - Discussão**: Escrever interpretação final

### Se ADOTAR γ=2:

**Arquivo:** `pesquisa_v7/scripts/020_train_adapter_solution.py`
- ✅ Já atualizado com `default=2`

**Arquivo:** `pesquisa_v7/README.md`
- Adicionar nota sobre γ=2 como configuração recomendada

**Arquivo:** `pesquisa_v7/ARQUITETURA_V7.md`
- Atualizar diagrama com γ=2

### Se MANTER γ=4:

**Arquivo:** `pesquisa_v7/scripts/020_train_adapter_solution.py`
- Reverter para `default=4`
- Documentar γ=2 como tentativa não bem-sucedida

---

## 7. Próximos Experimentos

### Se γ=2 foi bem-sucedido:
1. Aplicar γ=2 em Stage 3 (RECT e AB specialists)
2. Testar ensemble com γ=2
3. Documentar como best practice

### Se γ=2 falhou:
1. Investigar outras causas do underfitting:
   - BatchNorm distribution shift
   - Loss function inadequada
   - Features do Stage 1 não ótimas
2. Testar outras soluções:
   - Solution 2 (Ensemble)
   - Solution 3 (Hybrid)
   - Knowledge Distillation

---

## 8. Checklist de Análise Completa

- [ ] Verificar que treinamento terminou (arquivos .pt e .json existem)
- [ ] Extrair métricas principais (F1, gap, epochs)
- [ ] Comparar com baseline γ=4
- [ ] Gerar figuras de curvas de aprendizado
- [ ] Inspecionar checkpoint e confirmar parâmetros
- [ ] Aplicar critérios de decisão
- [ ] Atualizar documento 02_experimento_adapter_capacity.md
- [ ] Tomar decisão: adotar, considerar, ou rejeitar
- [ ] Atualizar scripts e documentação conforme decisão
- [ ] Planejar próximos experimentos
- [ ] Integrar resultados com tese (Capítulos 4, 5, 6)

---

**Última atualização**: 16/10/2025  
**Status**: 📋 Guia de análise pronto  
**Próximo passo**: Aguardar conclusão do treinamento e executar comandos deste guia
