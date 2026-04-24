import pandas as pd
import numpy as np
import os

CSV_FILE = '/mnt/SSD_SATA/dataset/dataset_index.csv'
TENSORS_FOLDER = "/mnt/SSD_SATA/dataset/Tensores_Treino/"

df_csv = pd.read_csv(CSV_FILE)

null_images = []
all_null = []
more_12_null = []

total_len = len(df_csv)

for l, row in enumerate(df_csv.itertuples(index=False)):
    print(f"{l}/{total_len} - Checking {row.name}...", end="\r")

    path_10 = os.path.join(TENSORS_FOLDER, f"{row.name}_10m.npy")
    path_20 = os.path.join(TENSORS_FOLDER, f"{row.name}_20m.npy")

    # safety: prevents crash if file does not exist
    if not os.path.exists(path_10) or not os.path.exists(path_20):
        continue

    t_10 = np.load(path_10)

    # vectorized (much faster than a loop)
    mask_null = (t_10 == 0).all(axis=tuple(range(1, t_10.ndim)))
    
    null_count = mask_null.sum()

    # save null indices
    idxs_null = np.where(mask_null)[0]
    null_images.extend([f"{row.name}_{i}" for i in idxs_null])

    # entirely null
    if null_count == t_10.shape[0]:
        all_null.append(row.name)

    # more than 12 nulls
    if null_count > 12:
        more_12_null.append(row.name)

print()

total_images = total_len * t_10.shape[0]

print(f"-> Total null images: {len(null_images)} ({len(null_images)/total_images:.2%})")
print(f"-> Total bases with all images null: {len(all_null)} ({len(all_null)/total_len:.2%})")
print(f"-> Total bases with more than 12 null images: {len(more_12_null)} ({len(more_12_null)/total_len:.2%})")

filtered_df = df_csv[df_csv['name'].isin(more_12_null)]

count_by_year = filtered_df['ano'].value_counts().sort_index()

print(count_by_year)