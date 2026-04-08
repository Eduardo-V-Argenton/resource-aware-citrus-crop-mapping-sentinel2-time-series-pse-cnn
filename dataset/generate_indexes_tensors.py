import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore", r"Mean of empty slice")

# =====================================================================
# CONFIGURAÇÕES DE DIRETÓRIO E PARÂMETROS
# =====================================================================
FICHEIRO_CSV = "dataset/dataset_index.csv"
PASTA_ORIGEM = "/mnt/SSD_SATA/Tensores_Treino/"
PASTA_DESTINO = "dataset/Tensores_Indexes/"

os.makedirs(PASTA_DESTINO, exist_ok=True)

h_alvo = 128
w_alvo = 128


def processar_poligono(nome_base):
    # Proteção essencial contra Deadlock no Multiprocessing
    torch.set_num_threads(1)

    caminho_saida = os.path.join(PASTA_DESTINO, f"{nome_base}_agro.npy")

    if os.path.exists(caminho_saida):
        return True

    try:
        # 1. Carrega e Escala
        t_10 = (
            np.load(os.path.join(PASTA_ORIGEM, f"{nome_base}_10m.npy")).astype(
                np.float32
            )
            / 10000.0
        )
        t_20 = (
            np.load(os.path.join(PASTA_ORIGEM, f"{nome_base}_20m.npy")).astype(
                np.float32
            )
            / 10000.0
        )

        _, _, h_10, w_10 = t_10.shape
        if h_10 < 15 or w_10 < 15:
            return False

        tensor_20_tmp = torch.from_numpy(t_20)
        t_20_zoom = F.interpolate(
            tensor_20_tmp, size=(h_10, w_10), mode="bilinear", align_corners=False
        ).numpy()

        # 2. Extração das Bandas Sentinel-2
        b2, b3, b4, b8 = t_10[:, 0], t_10[:, 1], t_10[:, 2], t_10[:, 3]
        b5, b11, b12 = t_20_zoom[:, 0], t_20_zoom[:, 4], t_20_zoom[:, 5]

        # 3. Os 5 Índices de Elite
        with np.errstate(divide="ignore", invalid="ignore"):
            mcari = np.where(b4 != 0, ((b5 - b4) - 0.2 * (b5 - b3)) * (b5 / b4), np.nan)
            ndre = np.where((b8 + b5) != 0, (b8 - b5) / (b8 + b5), np.nan)
            ndti = np.where((b11 + b12) != 0, (b11 - b12) / (b11 + b12), np.nan)
            sipi = np.where((b8 - b4) != 0, (b8 - b2) / (b8 - b4), np.nan)
            ci_green = np.where(b3 != 0, (b8 / b3) - 1, np.nan)

        # Forma do stack: (25 dias, 5 índices, H, W)
        tensor_indices = np.stack((mcari, ndre, ndti, sipi, ci_green), axis=1)

        # 4. Limpeza e Preenchimento Temporal (Essencial antes das estatísticas)
        tensor_indices = np.nan_to_num(
            tensor_indices, nan=np.nan, posinf=5.0, neginf=-5.0
        )
        tensor_indices = np.clip(tensor_indices, a_min=-5.0, a_max=5.0)

        soma_canais = np.sum(np.abs(tensor_indices), axis=1, keepdims=True)
        tensor_indices = np.where(soma_canais == 0.0, np.nan, tensor_indices)

        t, c, h, w = tensor_indices.shape
        flat_indices = tensor_indices.reshape(t, -1)

        mask_nans = np.isnan(flat_indices)
        for tempo in range(1, t):
            flat_indices[tempo] = np.where(
                mask_nans[tempo], flat_indices[tempo - 1], flat_indices[tempo]
            )

        mask_nans_pos = np.isnan(flat_indices)
        for tempo in range(t - 2, -1, -1):
            flat_indices[tempo] = np.where(
                mask_nans_pos[tempo], flat_indices[tempo + 1], flat_indices[tempo]
            )

        flat_indices = np.nan_to_num(flat_indices, nan=0.0)
        tensor_limpo = flat_indices.reshape(t, c, h, w)  # (25, 5, H, W)

        # =================================================================
        # 5. COMPRESSÃO AGRONÔMICA (Estatística Temporal)
        # =================================================================
        t_median = np.median(tensor_limpo, axis=0)
        t_10 = np.quantile(tensor_limpo, 0.1, axis=0)
        t_90 = np.quantile(tensor_limpo, 0.9, axis=0)
        t_auc = np.trapezoid(tensor_limpo, axis=0)

        # Empilha as 4 fotografias estatísticas
        # Resulta em 20 canais (5 índices x 4 estatísticas)
        t_agro = np.concatenate([t_median, t_10, t_90, t_auc], axis=0)

        mean = t_agro.mean(axis=(1, 2), keepdims=True)
        std = t_agro.std(axis=(1, 2), keepdims=True) + 1e-6

        t_agro = (t_agro - mean) / std

        # 6. Warp Espacial Direto para 128x128
        t_input = torch.from_numpy(t_agro).unsqueeze(0).float()  # (1, 20, H, W)

        t_resized = F.interpolate(
            t_input, size=(h_alvo, w_alvo), mode="bilinear", align_corners=False
        )

        t_final = t_resized.squeeze(0).numpy()  # Shape final limpo: (20, 128, 128)

        # Salva o tensor
        np.save(caminho_saida, t_final)
        return True

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    df_csv = pd.read_csv(FICHEIRO_CSV)
    df_csv = df_csv[df_csv["ano"] >= 2019]
    nomes_bases = df_csv["nome_base"].tolist()

    print(f"Iniciando Compressão Agronômica (Média, Desvio, Max, Min).")
    print(f"Formato de Saída: (20, 128, 128)")
    print(f"Total de polígonos na fila: {len(nomes_bases)}")

    nucleos = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=nucleos) as executor:
        futuros = {
            executor.submit(processar_poligono, nome): nome for nome in nomes_bases
        }
        for futuro in tqdm(
            as_completed(futuros),
            total=len(futuros),
            desc="Processando Tensores Estatísticos",
        ):
            resultado = futuro.result()
            if isinstance(resultado, str):
                print(f"Erro num polígono: {resultado}")

    print("\n[OK] Lote processado com velocidade máxima!")
