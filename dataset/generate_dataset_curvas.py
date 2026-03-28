import pandas as pd
import numpy as np
import os
import scipy.ndimage

FICHEIRO_CSV = 'dataset/dataset_index.csv'
PASTA_TENSORES = 'dataset/Tensores_Treino/'
PASTA_SAIDA_INDICES = 'dataset/Tensores_Indices_Calculados/'

os.makedirs(PASTA_SAIDA_INDICES, exist_ok=True)

df_csv = pd.read_csv(FICHEIRO_CSV)
df_csv = df_csv[df_csv['ano'] >= 2019]
dados_para_ml = []

for _, linha in df_csv.iterrows():
    nome_base = linha['nome_base']
    try:
        t_10 = np.load(os.path.join(PASTA_TENSORES, f"{nome_base}_10m.npy")).astype(np.float32)
        t_20 = np.load(os.path.join(PASTA_TENSORES, f"{nome_base}_20m.npy")).astype(np.float32)
        dias, bandas_10, h_10, w_10 = t_10.shape
        dias, bandas_20, h_20, w_20 = t_20.shape
        fator_h = h_10 / h_20
        fator_w = w_10 / w_20
        t_20 = scipy.ndimage.zoom(t_20, (1, 1, fator_h, fator_w), order=1)
    except Exception as e:
        print(f"Erro ao carregar os tensores para {nome_base}: {e}")
        continue

    b2 = t_10[:, 0, :, :]  # BLUE
    b4 = t_10[:, 2, :, :]  # RED
    b8 = t_10[:, 3, :, :]  # NIR
    
    b5 = t_20[:, 0, :, :]  # RE1
    b11 = t_20[:, 4, :, :]  # SWIR
    
    ndvi = np.divide(b8 - b4, b8 + b4, out=np.zeros_like(b8), where=(b8 + b4 != 0))
    evi = np.multiply(2.5, np.divide(b8 - b4, b8 + 6 * b4 - 7.5 * b2 + 1, out=np.zeros_like(b8), where=(b8 + 6 * b4 - 7.5 * b2 + 1 != 0)))
    ndre = np.divide(b8-b5, b8+b5, out=np.zeros_like(b8), where=(b8+b5 != 0))
    ndmi = np.divide(b8-b11, b8+b11, out=np.zeros_like(b8), where=(b8+b11 != 0))
    
    tensor_indices = np.stack((ndvi, evi, ndre, ndmi), axis=1)
    
    mediana = np.nanmedian(tensor_indices, axis=(2, 3)) 
    desvio = np.nanstd(tensor_indices, axis=(2, 3))     
    
    linha_mediana = mediana.flatten(order='F')
    linha_desvio = desvio.flatten(order='F')
    features_completas = np.concatenate((linha_mediana, linha_desvio))
    
    dados_para_ml.append({
        'nome_base': nome_base,
        'features': features_completas
    })
    
    print(f"Features extraídas com sucesso para: {nome_base}")

print("\nGerando DataFrame Final...")

indices = ["ndvi", "evi", "ndre", "ndmi"]

colunas_mediana = [
    f"{ind}_median_d{i}"
    for ind in indices
    for i in range(1, 26)
]
colunas_desvio = [
    f"{ind}_std_d{i}"
    for ind in indices
    for i in range(1, 26)
]
nomes_das_colunas = colunas_mediana + colunas_desvio

df_final_ml = pd.DataFrame([d['features'] for d in dados_para_ml], columns=nomes_das_colunas)

df_final_ml.insert(0, 'nome_base', [d['nome_base'] for d in dados_para_ml])

print(df_final_ml.head())

df_final_ml.to_csv('dataset/dataset_curvas.csv', index=False)
print("\nTabela 'dataset_curvas.csv' salva com sucesso! Pronta para o modelo.")
