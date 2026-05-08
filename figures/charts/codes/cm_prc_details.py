import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_recall_curve, auc

# =====================================================================
# 1. CARREGAMENTO DA MATRIZ DE CONFUSÃO (MÉDIA)
# =====================================================================
table_path = os.path.join("results", "recall_free", "paper_results", "consolidated_paper_table.csv")

try:
    df_table = pd.read_csv(table_path, index_col=0, header=[0, 1])
    TP = pd.to_numeric(df_table.loc['TP (True Pos)'], errors='coerce').mean()
    FN = pd.to_numeric(df_table.loc['FN (False Neg)'], errors='coerce').mean()
    FP = pd.to_numeric(df_table.loc['FP (False Pos)'], errors='coerce').mean()
    TN = pd.to_numeric(df_table.loc['TN (True Neg)'], errors='coerce').mean()
except Exception as e:
    print(f"Erro ao ler a tabela: {e}")
    exit()

# =====================================================================
# 2. CURVA PRECISION-RECALL (UNIFICADA E SUAVIZADA)
# =====================================================================
folder_path = os.path.join("results", "recall_free", "paper_results", "raw_predictions")
csv_files = glob.glob(os.path.join(folder_path, "predictions_year_*.csv"))

all_y_true, all_probs = [], []
for file in csv_files:
    df = pd.read_csv(file)
    all_y_true.extend(df['y_true'].values)
    all_probs.extend(df['model_probability'].values)

y_true_unified = np.array(all_y_true)
probs_unified = np.array(all_probs)

precision_vals, recall_vals, _ = precision_recall_curve(y_true_unified, probs_unified)
pr_auc = auc(recall_vals, precision_vals)
smoothed_precision = np.maximum.accumulate(precision_vals)

# =====================================================================
# 3. CÁLCULOS PARA A TABELA
# =====================================================================
P, N = TP + FN, TN + FP
Total = P + N
TPR, TNR = TP / P, TN / N
FPR, FNR = FP / N, FN / P
PPV = TP / (TP + FP)
ACC = (TP + TN) / Total
B_ACC = (TPR + TNR) / 2
F1 = 2 * (PPV * TPR) / (PPV + TPR)
den_mcc = float(TP + FP) * float(TP + FN) * float(TN + FP) * float(TN + FN)
MCC = ((TP * TN) - (FP * FN)) / np.sqrt(den_mcc) if den_mcc > 0 else 0.0

# =====================================================================
# 4. PLOTAGEM DO DASHBOARD (ESPAÇAMENTOS CORRIGIDOS)
# =====================================================================
plt.rcParams['font.family'] = 'serif'
fig = plt.figure(figsize=(18, 13)) 

gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.8], hspace=0.15, wspace=0.10)

ax_cm = fig.add_subplot(gs[0, 0])
cmap = ListedColormap(['#FFFFFF', '#B3CDE0', '#6497B1', '#005B96'])
ax_cm.imshow(np.array([[2, 0], [1, 3]]), cmap=cmap, aspect='auto')
ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(['Citrus', 'Non-Citrus'], fontsize=14)
ax_cm.set_yticklabels(['Citrus', 'Non-Citrus'], fontsize=14, rotation=90, va='center')
ax_cm.set_title('Average Confusion Matrix', fontsize=16, fontweight='bold', pad=15)

texts = [(0, 0, f"TP\n{TP:.1f}\n({(TP/Total)*100:.1f}%)", 'black'),
         (0, 1, f"FN\nType II Error\n{FN:.1f}\n({(FN/Total)*100:.1f}%)", 'black'),
         (1, 0, f"FP\nType I Error\n{FP:.1f}\n({(FP/Total)*100:.1f}%)", 'black'),
         (1, 1, f"TN\n{TN:.1f}\n({(TN/Total)*100:.1f}%)", 'white')]
for r, c, t, col in texts:
    ax_cm.text(c, r, t, ha='center', va='center', fontsize=15, color=col, fontweight='bold')

# --- PAINEL B: CURVA PR ---
ax_pr = fig.add_subplot(gs[0, 1])
ax_pr.plot(recall_vals, smoothed_precision, color='#D62728', linewidth=3, label=f'PSE-CNN (PR-AUC = {pr_auc:.3f})')
baseline = np.sum(y_true_unified) / len(y_true_unified)
ax_pr.axhline(y=baseline, color='gray', linestyle='--', linewidth=2, label=f'Random Guess ({baseline:.2f})')
ax_pr.set_xlim([-0.02, 1.02])
ax_pr.set_ylim([0.0, 1.05])
ax_pr.set_xlabel('Recall (Sensitivity)', fontsize=14)
ax_pr.set_ylabel('Precision (PPV)', fontsize=14)
ax_pr.set_title('Unified Precision-Recall Curve', fontsize=16, fontweight='bold', pad=15)
ax_pr.legend(loc='lower left', fontsize=14)
ax_pr.grid(True, linestyle='--', alpha=0.5)

# --- PAINEL C: TABELA DE MÉTRICAS ---
ax_table = fig.add_subplot(gs[1, :])
ax_table.axis('off')

columns = ["Metric", "Value", "Ideal", "Formula", "What it means", "Error Type Focus"]
table_data = [
    ["TPR (Recall)", f"{TPR:.3f}", "High (↑)", "TP / (TP+FN)", "Detect real citrus", "Type II ↓"],
    ["TNR (Specificity)", f"{TNR:.3f}", "High (↑)", "TN / (TN+FP)", "Reject non-citrus", "Type I ↓"],
    ["FPR", f"{FPR:.3f}", "Low (↓)", "FP / (TN+FP)", "False alarms", "Type I ↓"],
    ["FNR", f"{FNR:.3f}", "Low (↓)", "FN / (TP+FN)", "Missed citrus", "Type II ↓"],
    ["PPV (Precision)", f"{PPV:.3f}", "High (↑)", "TP / (TP+FP)", "Trust citrus alerts", "—"],
    ["Accuracy", f"{ACC:.3f}", "High (↑)", "(TP+TN) / Total", "Overall correctness", "—"],
    ["Balanced Accuracy", f"{B_ACC:.3f}", "High (↑)", "(TPR+TNR) / 2", "Imbalance-safe acc.", "—"],
    ["F1-score", f"{F1:.3f}", "High (↑)", "2TP / (2TP+FP+FN)", "Precision-Recall tradeoff", "—"],
    ["MCC", f"{MCC:.3f}", "High (↑)", r"$\frac{(TP \times TN - FP \times FN)}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$", "Correlation quality", "—"]
]

col_widths = [0.14, 0.08, 0.08, 0.35, 0.19, 0.16]

# Tabela ocupa seu próprio espaço perfeitamente
table = ax_table.table(
    cellText=table_data, 
    colLabels=columns, 
    loc='center', 
    cellLoc='center', 
    colWidths=col_widths, 
    bbox=[0, 0, 1, 1] 
)

table.auto_set_font_size(False)
table.set_fontsize(11) 
table.scale(1, 1.8)

for j, key in enumerate(columns):
    cell = table[0, j]
    cell.set_text_props(weight='bold')
    cell.set_facecolor('#EAEAEA')

last_col_idx = len(columns) - 1

for i in range(len(table_data) + 1):
    cell = table[i, last_col_idx]
    cell.set_text_props(wrap=True)
    for j in range(len(columns)):
        cell = table[i, j]
        if i == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(weight='bold', color='white', fontsize=14)
        else:
            cell.set_facecolor('#F8F9FA' if i % 2 != 0 else '#FFFFFF')

plt.tight_layout(pad=2)
plt.savefig("figures/charts/exported/comprehensive_metrics_dashboard.png", dpi=650, bbox_inches='tight')
plt.show()
