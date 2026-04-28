import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# =====================================================================
# CONFIGURAÇÕES ACADÊMICAS
# =====================================================================
results_config = {
    "paper_results": "PSE-CNN (Proposed)",
    "paper_results_pse_tae": "PSE-TAE",
    "paper_results_pse_transformer": "PSE-Transformer",
    "paper_results_rf": "RF (Bands)",
    "paper_results_rf_bands_indexes": "RF (Bands+Idx)",
    "paper_results_xgb": "XGB (Bands)",
    "paper_results_xgb_bands_indexes": "XGB (Bands+Idx)"
}

styles = {
    "PSE-CNN (Proposed)":   {"color": "#D62728"}, # Vermelho Forte
    "PSE-TAE":              {"color": "#8C564B"}, #Marrom
    "PSE-Transformer":      {"color": "#17BECF"}, #Ciano
    "RF (Bands)":           {"color": "#1F77B4"}, # Azul
    "RF (Bands+Idx)":       {"color": "#2CA02C"}, # Verde
    "XGB (Bands)":          {"color": "#FF7F0E"}, # Laranja
    "XGB (Bands+Idx)":      {"color": "#9467BD"},  # Roxo
}

# =====================================================================
# CARREGAMENTO DOS DADOS
# =====================================================================
data_list = []

for folder_path, display_name in results_config.items():
    pattern = os.path.join("results", "recall_free", folder_path, "classification_reports", "test_year_*.csv")
    files = glob.glob(pattern)

    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            parts = filename.replace(".csv", "").split("_")
            year = int(parts[2])
            seed = int(parts[4])

            df_report = pd.read_csv(file_path, index_col=0)
            df_report.index = df_report.index.astype(str)

            for label in ['1', '1.0', 'Class 1']:
                if label in df_report.index:
                    row = df_report.loc[label]
                    data_list.append({
                        "Architecture": display_name,
                        "Year": year,
                        "Seed": seed,
                        "Precision": row['precision'],
                        "Recall": row['recall'],
                        "F1-Score": row['f1-score']
                    })
                    break
        except Exception as e:
            print(f"Error: {e}")

df_results = pd.DataFrame(data_list)

# =====================================================================
# PLOTAGEM DE ALTO IMPACTO (BOXPLOT UNIFICADO + SCATTER ALINHADO)
# =====================================================================
plt.rcParams['font.family'] = 'serif'

fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.2)

ax_f1 = fig.add_subplot(gs[0, :])
ax_prec = fig.add_subplot(gs[1, 0])
ax_rec = fig.add_subplot(gs[1, 1])

metrics = [
    ("F1-Score", ax_f1, "(a)"),
    ("Precision", ax_prec, "(b)"),
    ("Recall", ax_rec, "(c)")
]

models = list(results_config.values())

for metric_name, ax, label in metrics:
    
    box_data = []
    positions = np.arange(1, len(models) + 1)
    
    for j, model_name in enumerate(models):
        model_data = df_results[df_results["Architecture"] == model_name]
        data_values = model_data[metric_name].dropna().values
        box_data.append(data_values)
        
        pos = positions[j]
        st = styles[model_name]

        # CORREÇÃO 1 e 2: Sem jitter (posição X exata) e usando a cor do modelo
        if len(data_values) > 0:
            x_positions = [pos] * len(data_values)
            ax.scatter(x_positions, data_values, color=st["color"], alpha=0.6, 
                       s=35, zorder=5, edgecolors='white', linewidths=0.8)

    # Criando o Boxplot
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False, # Não mostrar outliers padrões
        zorder=3
    )
    
    # Customização de Cores do Boxplot
    for item_name in ['boxes', 'whiskers', 'caps', 'medians']:
        for j, element in enumerate(bp[item_name]):
            model_idx = j if item_name in ['boxes', 'medians'] else j // 2
            color = styles[models[model_idx]]["color"]
            
            if item_name == 'boxes':
                element.set_facecolor(color)
                element.set_alpha(0.3)
                element.set_edgecolor(color)
                element.set_linewidth(1.5)
            elif item_name == 'medians':
                element.set_color('black') 
                element.set_linewidth(2.5)
            else:
                element.set_color(color)
                element.set_linewidth(1.5)

    # =========================
    # Eixos, Ticks e Escalas
    # =========================
    ax.set_ylim(0.0, 1) 
    ax.set_yticks(np.arange(0.0, 1.0, 0.1))
    
    ax.set_xticks(positions)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0) 
    
    ax.set_xlim(0.3, len(models) + 0.7)

    # =========================
    # Estética High-Impact
    # =========================
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.tick_params(axis='y', which='major', direction='in', length=5, width=1, labelsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False) 
    ax.spines['left'].set_linewidth(1.0)

    ax.text(0.01, 0.96, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

# =====================================================================
# LEGENDA GLOBAL
# =====================================================================
legend_handles = []
for model_name in models:
    color = styles[model_name]["color"]
    patch = mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.6, label=model_name, linewidth=1.5)
    legend_handles.append(patch)

fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.06), 
    ncol=4,
    frameon=False, 
    fontsize=13
)

plt.tight_layout(rect=[0, 0.12, 1, 0.98])

output_dir = "figures/charts/exported"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "classification_report.png")

plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()