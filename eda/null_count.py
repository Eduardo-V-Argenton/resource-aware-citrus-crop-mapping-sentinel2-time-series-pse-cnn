import pandas as pd
import numpy as np
import os

FICHEIRO_CSV = '/mnt/SSD_SATA/dataset/dataset_index.csv'
PASTA_TENSORES = "/mnt/SSD_SATA/dataset/Tensores_Treino/"

df_csv = pd.read_csv(FICHEIRO_CSV)

null_images = []
all_null = []
more_12_null = []

len_total = len(df_csv)

for l, linha in enumerate(df_csv.itertuples(index=False)):
    print(f"{l}/{len_total} - Verificando {linha.name}...", end="\r")

    path_10 = os.path.join(PASTA_TENSORES, f"{linha.name}_10m.npy")
    path_20 = os.path.join(PASTA_TENSORES, f"{linha.name}_20m.npy")

    # segurança: evita crash se arquivo não existir
    if not os.path.exists(path_10) or not os.path.exists(path_20):
        continue

    t_10 = np.load(path_10)

    # vetoriza (muito mais rápido que loop)
    mask_null = (t_10 == 0).all(axis=tuple(range(1, t_10.ndim)))
    
    null_count = mask_null.sum()

    # salva índices nulos
    idxs_null = np.where(mask_null)[0]
    null_images.extend([f"{linha.name}_{i}" for i in idxs_null])

    # tudo nulo
    if null_count == t_10.shape[0]:
        all_null.append(linha.name)

    # mais de 12 nulos
    if null_count > 12:
        more_12_null.append(linha.name)

print()

total_imagens = len_total * t_10.shape[0]

print(f"-> Total de imagens nulas: {len(null_images)} ({len(null_images)/total_imagens:.2%})")
print(f"-> Total de bases com todas as imagens nulas: {len(all_null)} ({len(all_null)/len_total:.2%})")
print(f"-> Total de bases com mais de 12 imagens nulas: {len(more_12_null)} ({len(more_12_null)/len_total:.2%})")

df_filtrado = df_csv[df_csv['name'].isin(more_12_null)]

count_por_ano = df_filtrado['ano'].value_counts().sort_index()

print(count_por_ano)