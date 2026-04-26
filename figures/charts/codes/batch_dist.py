import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm

# =====================================================================
# CONFIGURATIONS
# =====================================================================
INDEX_FILE = '/mnt/SSD_SATA/dataset/dataset_index.csv'
BATCH_TRACKING_DIR = 'results/recall_free/paper_results/batch_tracking/' 
TARGET_COLUMN = 'label_ia'

print("Loading dataset index to map image names to classes...")
# Load index and create a fast lookup dictionary: {'img_001': 0, 'img_002': 1, ...}
df_index = pd.read_csv(INDEX_FILE)

# Clean the base_name just in case it has suffixes in the tracking file
df_index['clean_name'] = df_index['name'].str.replace(r"_p\d+", "", regex=True)
class_mapping = dict(zip(df_index['name'], df_index[TARGET_COLUMN]))

# =====================================================================
# PROCESSING BATCH FILES
# =====================================================================
batch_files = glob(os.path.join(BATCH_TRACKING_DIR, "*.csv"))

if not batch_files:
    raise FileNotFoundError(f"No CSV files found in {BATCH_TRACKING_DIR}")

print(f"Found {len(batch_files)} batch tracking files. Processing...")

batch_data = []

for file in batch_files:
    df_batch = pd.read_csv(file)
    
    # Iterate through every row (every batch)
    for _, row in tqdm(df_batch.iterrows(), total=len(df_batch), desc=f"Reading {os.path.basename(file)}", leave=False):
        samples = row['samples'].split('|')
        
        class_0_count = 0
        class_1_count = 0
        
        for sample in samples:
            # Map the sample string to its class
            sample_class = class_mapping.get(sample, None)
            
            if sample_class == 0:
                class_0_count += 1
            elif sample_class == 1:
                class_1_count += 1
                
        batch_data.append({
            'Phase': row['phase'].capitalize(), # 'Train' or 'Val'
            'Non-Citrus (0)': class_0_count,
            'Citrus (1)': class_1_count
        })

df_results = pd.DataFrame(batch_data)

# Convert to Long Format for calculating medians
df_melted = df_results.melt(id_vars=['Phase'], 
                            value_vars=['Non-Citrus (0)', 'Citrus (1)'], 
                            var_name='Class', 
                            value_name='Samples per Batch')

# =====================================================================
# CALCULATING MEDIANS
# =====================================================================
# Calculate the median for each Phase and Class
medians = df_melted.groupby(['Phase', 'Class'])['Samples per Batch'].median().reset_index()

print("\n" + "="*40)
print(" MEDIAN SAMPLES PER BATCH")
print("="*40)
print(medians.to_string(index=False))
print("="*40)

# =====================================================================
# PLOTTING (ACADEMIC HIGH-IMPACT STYLE)
# =====================================================================
# Set a clean, academic style and match the font to previous charts
plt.rcParams['font.family'] = 'serif'
sns.set_theme(style="white", context="paper", font_scale=1.2) # Alterado para 'white' para controlar o grid manualmente

fig, ax = plt.subplots(figsize=(8, 6))

# Choose colors that are colorblind-friendly and distinct
palette = {"Non-Citrus (0)": "#1F77B4", "Citrus (1)": "#FF7F0E"}
hue_order = ["Non-Citrus (0)", "Citrus (1)"] # Garante ordem lógica no gráfico

# Plot the bar chart
sns.barplot(
    data=medians, 
    x='Phase', 
    y='Samples per Batch', 
    hue='Class', 
    hue_order=hue_order,
    palette=palette,
    edgecolor='#333333', # Cinza escuro elegante em vez de preto chapado
    alpha=0.85,          # Opacidade alta para dar "peso" às barras
    linewidth=1.5,
    ax=ax
)

# =========================
# FIXING THE OVERLAP
# =========================
# Calcula a barra mais alta e adiciona 25% de espaço no topo
max_val = medians['Samples per Batch'].max()
ax.set_ylim(0, max_val * 1.25)

# Customize Labels and Title
ax.set_xlabel('Dataset Phase', fontsize=13, fontweight='bold')
ax.set_ylabel('Median Number of Samples per Batch', fontsize=13, fontweight='bold')

# Add the exact median numbers on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.0f', padding=4, fontsize=12, fontweight='bold', color='#333333')

# =========================
# AESTHETICS & LEGEND
# =========================
# Legenda horizontal alinhada ao centro superior, combinando com seus outros gráficos
ax.legend(title='', loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=12)

# Grid apenas horizontal e muito discreto
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.tick_params(axis='both', which='major', direction='in', length=5, width=1, labelsize=12)

# Remove top and right borders for a cleaner look
sns.despine()

# Save the plot in high resolution for the paper
os.makedirs("figures/charts/exported", exist_ok=True)
output_image_path = "figures/charts/exported/median_batch_composition.png"

plt.tight_layout()
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')

print(f"\nPlot successfully saved as '{output_image_path}'")
plt.show()