import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Configurações de caminho (iguais aos seus scripts)
FICHEIRO_CSV = '/mnt/SSD_SATA/dataset/dataset_index.csv'
PASTA_ORIGEM = "/mnt/SSD_SATA/dataset/Tensores_Treino/"

# Lê o CSV
df_csv = pd.read_csv(FICHEIRO_CSV)
df_csv = df_csv[df_csv['ano'] >= 2019]

alturas = []
larguras = []

print("Analisando as dimensões reais dos seus talhões. Isso será bem rápido...")

for _, linha in tqdm(df_csv.iterrows(), total=len(df_csv), desc="Lendo shapes"):
    nome_base = linha['name']
    caminho_10m = os.path.join(PASTA_ORIGEM, f"{nome_base}_10m.npy")
    
    if not os.path.exists(caminho_10m):
        continue

    try:
        # mmap_mode='r' lê apenas os metadados do arquivo (super rápido e leve)
        tensor = np.load(caminho_10m, mmap_mode='r')
        
        # O shape do seu tensor original é (25, 4, H, W)
        _, _, h, w = tensor.shape
        
        alturas.append(h)
        larguras.append(w)
        
    except Exception as e:
        print(f"Erro ao ler {nome_base}: {e}")

# Transforma em arrays numpy para facilitar a matemática
alturas = np.array(alturas)
larguras = np.array(larguras)

# Imprime o Laudo Estatístico
print("\n" + "="*50)
print(" RELATÓRIO ESPACIAL DO DATASET (Pixels a 10m)")
print("="*50)
print(f"Total de polígonos analisados: {len(alturas)}")
print("-" * 50)
print(f"ALTURA (H):")
print(f"  Mínima:  {np.min(alturas)}")
print(f"  Média:   {np.mean(alturas):.0f}")
print(f"  Mediana: {np.median(alturas):.0f}")
print(f"  Máxima:  {np.max(alturas)}")
print("-" * 50)
print(f"LARGURA (W):")
print(f"  Mínima:  {np.min(larguras)}")
print(f"  Média:   {np.mean(larguras):.0f}")
print(f"  Mediana: {np.median(larguras):.0f}")
print(f"  Máxima:  {np.max(larguras)}")
print("=" * 50)