import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Path configurations (same as your scripts)
CSV_FILE = '/mnt/SSD_SATA/dataset/dataset_index.csv'
SOURCE_FOLDER = "/mnt/SSD_SATA/dataset/Tensores_Treino/"

# Read the CSV
df_csv = pd.read_csv(CSV_FILE)
df_csv = df_csv[df_csv['ano'] >= 2019]

heights = []
widths = []

print("Analyzing the actual dimensions of your fields. This will be very fast...")

for _, row in tqdm(df_csv.iterrows(), total=len(df_csv), desc="Reading shapes"):
    base_name = row['name']
    path_10m = os.path.join(SOURCE_FOLDER, f"{base_name}_10m.npy")
    
    if not os.path.exists(path_10m):
        continue

    try:
        # mmap_mode='r' reads only the file metadata (super fast and lightweight)
        tensor = np.load(path_10m, mmap_mode='r')
        
        # Your original tensor shape is (25, 4, H, W)
        _, _, h, w = tensor.shape
        
        heights.append(h)
        widths.append(w)
        
    except Exception as e:
        print(f"Error reading {base_name}: {e}")

# Convert to numpy arrays for easier math
heights = np.array(heights)
widths = np.array(widths)

# Print the Statistical Report
print("\n" + "="*50)
print(" DATASET SPATIAL REPORT (10m Pixels)")
print("="*50)
print(f"Total polygons analyzed: {len(heights)}")
print("-" * 50)
print("HEIGHT (H):")
print(f"  Min:    {np.min(heights)}")
print(f"  Mean:   {np.mean(heights):.0f}")
print(f"  Median: {np.median(heights):.0f}")
print(f"  Max:    {np.max(heights)}")
print("-" * 50)
print("WIDTH (W):")
print(f"  Min:    {np.min(widths)}")
print(f"  Mean:   {np.mean(widths):.0f}")
print(f"  Median: {np.median(widths):.0f}")
print(f"  Max:    {np.max(widths)}")
print("=" * 50)