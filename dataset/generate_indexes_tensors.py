import pandas as pd
import numpy as np
import os
import scipy.ndimage
import warnings
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings(action='ignore', category=UserWarning, module='scipy.ndimage')

FICHEIRO_CSV = 'dataset/dataset_index.csv'
PASTA_ORIGEM = 'dataset/Tensores_Treino/'
PASTA_DESTINO = 'dataset/Tensores_Indexes_FFT/'

os.makedirs(PASTA_DESTINO, exist_ok=True)

df_csv = pd.read_csv(FICHEIRO_CSV)
df_csv = df_csv[df_csv['ano'] >= 2019]

dias_totais = 25

h_alvo = 128
w_alvo = 128

print(f"Iniciando pré-processamento de {len(df_csv)} polígonos (Alternativa A: Interpolação para {h_alvo}x{w_alvo})...")

for _, linha in tqdm(df_csv.iterrows(), total=len(df_csv), desc="Processando FFT e Warp"):
    nome_base = linha['nome_base']
    caminho_saida = os.path.join(PASTA_DESTINO, f"{nome_base}_fft.npy")
    
    if os.path.exists(caminho_saida):
        continue

    try:
        # 1. Carrega e Escala
        t_10 = np.load(os.path.join(PASTA_ORIGEM, f"{nome_base}_10m.npy")).astype(np.float32) / 10000.0
        t_20 = np.load(os.path.join(PASTA_ORIGEM, f"{nome_base}_20m.npy")).astype(np.float32) / 10000.0
        
        # 2. Zoom Dinâmico na 20m para igualar o tamanho original da 10m
        _, _, h_10, w_10 = t_10.shape
        
        if h_10 < 15 or w_10 < 15:
            continue
            
        _, _, h_20, w_20 = t_20.shape
        fator_h, fator_w = h_10 / h_20, w_10 / w_20
        t_20_zoom = scipy.ndimage.zoom(t_20, (1, 1, fator_h, fator_w), order=1)[:, :, :h_10, :w_10]
        
        # 3. Separação de Bandas para Índices
        b2 = t_10[:, 0, :, :]   # BLUE
        b4 = t_10[:, 2, :, :]   # RED
        b8 = t_10[:, 3, :, :]   # NIR
        b5 = t_20_zoom[:, 0, :, :]   # RE1
        b11 = t_20_zoom[:, 4, :, :]  # SWIR
        
        # 4. Cálculo de Índices Pixel a Pixel (Shape: 25, H, W)
        msavi2 = (2 * b8 + 1 - np.sqrt(np.maximum((2 * b8 + 1)**2 - 8 * (b8 - b4), 0))) / 2.0
        evi = np.multiply(2.5, np.divide(b8 - b4, b8 + 6 * b4 - 7.5 * b2 + 1, out=np.zeros_like(b8), where=(b8 + 6 * b4 - 7.5 * b2 + 1 != 0)))
        ndre = np.divide(b8 - b5, b8 + b5, out=np.zeros_like(b8), where=(b8 + b5 != 0))
        ndmi = np.divide(b8 - b11, b8 + b11, out=np.zeros_like(b8), where=(b8 + b11 != 0))
        
        # Junta os índices criando o eixo do canal (Shape: 25, 4, H, W)
        tensor_indices = np.stack((msavi2, evi, ndre, ndmi), axis=1)
        
        # O Canal Âncora: Adicionamos a Banda B8 bruta
        b8_expandido = np.expand_dims(b8, axis=1)
        
        # Tensor com 5 variáveis: 4 índices + B8
        tensor_completo = np.concatenate((tensor_indices, b8_expandido), axis=1)
        
        # 6. Transformada de Fourier (FFT) no Eixo do Tempo (axis=0)
        fft_vals = np.fft.fft(tensor_completo, axis=0)
        
        mean_val = np.real(fft_vals[0]) / dias_totais
        amp_val = np.abs(fft_vals[1]) / (dias_totais / 2.0)
        phase_val = np.angle(fft_vals[1])
        
        # Empilha os 3 componentes (5 vars * 3 componentes = 15 Canais)
        tensor_fft = np.concatenate([mean_val, amp_val, phase_val], axis=0).astype(np.float32)
        
        # -----------------------------------------------------------------
        # ALTERNATIVA A: INTERPOLAÇÃO / WARP ESPACIAL (Sem fundo preto)
        # -----------------------------------------------------------------
        # O interpolate do PyTorch exige formato 4D: (Batch, Canais, H, W)
        # Usamos unsqueeze(0) para simular o Batch = 1
        t_input = torch.from_numpy(tensor_fft).unsqueeze(0)
        
        # Estica ou encolhe perfeitamente a imagem para 64x64 usando Bilinear
        t_resized = F.interpolate(t_input, size=(h_alvo, w_alvo), mode='bilinear', align_corners=False)
        
        # Remove a dimensão falsa do Batch com squeeze(0)
        t_final = t_resized.squeeze(0)
        
        # Salva o tensor final (Shape: 15, 64, 64)
        np.save(caminho_saida, t_final.numpy())
        
    except Exception as e:
        print(f"\nErro no polígono {nome_base}: {e}")

print(f"\nArquivos FFT gerados e padronizados para {h_alvo}x{w_alvo} com sucesso!")