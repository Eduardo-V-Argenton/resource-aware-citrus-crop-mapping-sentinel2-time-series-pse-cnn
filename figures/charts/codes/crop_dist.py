import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
FILE_PATH = 'results/class_distribution.csv'

COLOR_CITRUS = '#FF7F0E' 
COLOR_OTHERS = '#D3D3D3' 
EDGE_COLOR = '#333333'

CLASSES_TO_REMOVE = ['false', 'False', 'removed_small', 'Removed Small']

# =====================================================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =====================================================================
df = pd.read_csv(FILE_PATH)
df = df[~df['class'].isin(CLASSES_TO_REMOVE)].copy()

total_samples = df['count'].sum()
df['percentage'] = df['count'] / total_samples

df['class_display'] = df['class'].str.replace('_', ' ').str.title()
df = df.sort_values(by='percentage', ascending=True).reset_index(drop=True)

colors = [COLOR_CITRUS if cls.lower() == 'citrus' else COLOR_OTHERS for cls in df['class']]

# =====================================================================
# PLOTAGEM (100% FOCADA EM PORCENTAGEM)
# =====================================================================
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(10, 6))

# Agora passamos a 'percentage' para o tamanho da barra, não mais o 'count'
bars = ax.barh(
    df['class_display'], 
    df['percentage'], 
    color=colors, 
    edgecolor=EDGE_COLOR, 
    linewidth=1.2,
    alpha=0.9,
    height=0.65
)

# =========================
# RÓTULOS E LIMITES
# =========================
max_pct = df['percentage'].max()
ax.set_xlim(0, max_pct * 1.15) # Dá espaço para o texto respirar no final

for bar, pct in zip(bars, df['percentage']):
    ax.text(
        bar.get_width() + (max_pct * 0.015), 
        bar.get_y() + bar.get_height() / 2,    
        f'{pct * 100:.1f}%', 
        ha='left', va='center', 
        fontsize=12, fontweight='bold', color=EDGE_COLOR
    )

# =========================
# ESTÉTICA E EIXOS
# =========================
ax.set_xlabel('Percentage of samples', fontsize=13, fontweight='bold', labelpad=10)

# Transforma os números do eixo X (0.1, 0.2) em formato de porcentagem (10%, 20%)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.yticks(fontsize=12) 
plt.xticks(fontsize=11)

ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
ax.tick_params(axis='both', which='major', direction='in', length=5, width=1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)

# =====================================================================
# SALVAR E MOSTRAR
# =====================================================================
output_dir = "figures/charts/exported"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "class_distribution.png")

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"Gráfico atualizado salvo em: {output_path}")
plt.show()