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
FICHEIRO_CSV = '/mnt/SSD_SATA/dataset/dataset_index.csv'
PASTA_ORIGEM = "/mnt/SSD_SATA/dataset/Tensores_Treino/"
PASTA_DESTINO = "dataset/Tensors/"

os.makedirs(PASTA_DESTINO, exist_ok=True)


# =====================================================================
# FUNÇÃO AUXILIAR: PROCESSAMENTO BASE
# =====================================================================
def carregar_e_limpar_tensor(nome_base):
    """Carrega as bandas, faz upsample, clip e gap-filling. Retorna tensor com NaNs no fundo."""
    t_10 = (
        np.load(os.path.join(PASTA_ORIGEM, f"{nome_base}_10m.npy")).astype(np.float32)
        / 10000.0
    )
    t_20 = (
        np.load(os.path.join(PASTA_ORIGEM, f"{nome_base}_20m.npy")).astype(np.float32)
        / 10000.0
    )

    _, _, h_10, w_10 = t_10.shape

    # Upsample das bandas de 20m (Nearest Neighbor)
    t_20_tensor = torch.from_numpy(t_20)
    t_20_up = F.interpolate(t_20_tensor, size=(h_10, w_10), mode="nearest").numpy()

    # Extração e Empilhamento
    b2, b3, b4, b8 = t_10[:, 0], t_10[:, 1], t_10[:, 2], t_10[:, 3]
    b5, b6, b7, b8a, b11, b12 = (
        t_20_up[:, 0],
        t_20_up[:, 1],
        t_20_up[:, 2],
        t_20_up[:, 3],
        t_20_up[:, 4],
        t_20_up[:, 5],
    )
    tensor_base = np.stack((b2, b3, b4, b8, b5, b6, b7, b8a, b11, b12), axis=1)

    # Limpeza e Clip (-2.0 a 2.0)
    tensor_base = np.nan_to_num(tensor_base, nan=np.nan, posinf=2.0, neginf=-2.0)
    tensor_base = np.clip(tensor_base, a_min=-2.0, a_max=2.0)
    
    mascara_fazenda = (tensor_base[0, 0, :, :] != 0.0) & (tensor_base[0, 0, :, :] > -0.5)
    flat_base = tensor_base[:, :, mascara_fazenda]
    t, c, p = flat_base.shape
    if p < 32:
        return None
        
    mask_nans = np.isnan(flat_base) | (flat_base == 0.0)
    for tempo in range(1, t):
        flat_base[tempo] = np.where(
            mask_nans[tempo], flat_base[tempo - 1], flat_base[tempo]
        )

    mask_nans_pos = np.isnan(flat_base) | (flat_base == 0.0)
    for tempo in range(t - 2, -1, -1):
        flat_base[tempo] = np.where(
            mask_nans_pos[tempo], flat_base[tempo + 1], flat_base[tempo]
        )

    return flat_base


# =====================================================================
# PROCESSAMENTO E SALVAMENTO (Sem Normalização)
# =====================================================================
def gerar_dataset_final(nome_base):
    torch.set_num_threads(1)
    caminho_saida = os.path.join(PASTA_DESTINO, f"{nome_base}_pse.npy")

    # Sistema de Autocura (Pula se já existe)
    if os.path.exists(caminho_saida):
        return True

    try:
        tensor = carregar_e_limpar_tensor(nome_base)
        if tensor is None:
            return None

        # Zera os NaNs (background) para o formato final da Rede Neural
        # Salva mantendo os valores originais em reflectância
        tensor_final = np.nan_to_num(tensor, nan=0.0).astype(np.float16)

        # Salva o ÚNICO arquivo gerado direto na pasta final
        np.save(caminho_saida, tensor_final)
        return True
    except Exception as e:
        return f"Erro em {nome_base}: {str(e)}"


# =====================================================================
# EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    df_csv = pd.read_csv(FICHEIRO_CSV)

    nomes_bases = df_csv["name"].tolist()
    nucleos = max(1, os.cpu_count() - 2)

    print(f"\n--- Gerando Dataset Final (Limpeza e Gap Filling) ---")
    
    # Processamento paralelo com barra de progresso
    with ProcessPoolExecutor(max_workers=nucleos) as executor:
        futuros = {
            executor.submit(gerar_dataset_final, nome): nome
            for nome in nomes_bases
        }

        for futuro in tqdm(as_completed(futuros), total=len(futuros)):
            resultado = futuro.result()
            if isinstance(resultado, str):  # Se for uma string, é a mensagem de erro
                print(resultado)

    print("\nProcessamento direto concluído! Arquivos prontos para o PyTorch.")
