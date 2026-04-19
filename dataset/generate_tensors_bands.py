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
# FUNÇÃO AUXILIAR: PROCESSAMENTO BASE (Usada nas duas passadas)
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
    if h_10 < 15 or w_10 < 15:
        return None

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

    # Mascara o background (transforma 0 absoluto em NaN)
    soma_canais = np.sum(np.abs(tensor_base), axis=1, keepdims=True)
    tensor_base = np.where(soma_canais == 0.0, np.nan, tensor_base)

    t, c, h, w = tensor_base.shape
    flat_base = tensor_base.reshape(t, c, h * w)

    # Preenchimento Temporal (Gap Filling)
    mask_nans = np.isnan(flat_base)
    for tempo in range(1, t):
        flat_base[tempo] = np.where(
            mask_nans[tempo], flat_base[tempo - 1], flat_base[tempo]
        )

    mask_nans_pos = np.isnan(flat_base)
    for tempo in range(t - 2, -1, -1):
        flat_base[tempo] = np.where(
            mask_nans_pos[tempo], flat_base[tempo + 1], flat_base[tempo]
        )

    return flat_base  # Retorna o tensor na RAM (não salva no disco!)


# =====================================================================
# PASSADA 1: CALCULAR ESTATÍSTICAS (Sem salvar no disco)
# =====================================================================
def calcular_estatisticas(nome_base):
    torch.set_num_threads(1)
    try:
        flat_base = carregar_e_limpar_tensor(nome_base)
        if flat_base is None:
            return None

        somas = np.nansum(flat_base, axis=(0, 2))
        somas_sq = np.nansum(flat_base**2, axis=(0, 2))
        contagem = np.sum(~np.isnan(flat_base), axis=(0, 2))

        # O Python descarta flat_base da RAM ao sair da função. Zero disco usado.
        return {"somas": somas, "somas_sq": somas_sq, "contagem": contagem}
    except Exception as e:
        return f"Erro Fase 1 em {nome_base}: {str(e)}"


# =====================================================================
# PASSADA 2: PADRONIZAR E SALVAR (Z-Score)
# =====================================================================
def gerar_dataset_final(nome_base, media_global, std_global):
    torch.set_num_threads(1)
    caminho_saida = os.path.join(PASTA_DESTINO, f"{nome_base}_pse.npy")

    if os.path.exists(caminho_saida):
        return True

    try:
        # Refazemos o processamento inicial na RAM
        tensor = carregar_e_limpar_tensor(nome_base)
        if tensor is None:
            return None

        # Reshape para o cálculo vetorizado do Z-Score
        media = media_global.reshape(1, 10, 1)
        std = std_global.reshape(1, 10, 1)

        # Aplica a padronização apenas onde há plantação (ignora NaNs)
        tensor_padronizado = np.where(~np.isnan(tensor), (tensor - media) / std, np.nan)

        # Zera os NaNs (background) para o formato final da Rede Neural
        tensor_final = np.nan_to_num(tensor_padronizado, nan=0.0).astype(np.float16)

        # Salva o ÚNICO arquivo gerado direto na pasta final
        np.save(caminho_saida, tensor_final)
        return True
    except Exception as e:
        return f"Erro Fase 2 em {nome_base}: {str(e)}"


# =====================================================================
# EXECUÇÃO PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    df_csv = pd.read_csv(FICHEIRO_CSV)
    df_csv = df_csv[df_csv["ano"] >= 2019]

    nomes_bases = df_csv["name"].tolist()
    nucleos = max(1, os.cpu_count() - 2)

    soma_total = np.zeros(10, dtype=np.float64)
    soma_sq_total = np.zeros(10, dtype=np.float64)
    contagem_total = np.zeros(10, dtype=np.float64)

    print(f"\n--- PASSADA 1: Calculando Estatísticas (Apenas Memória RAM) ---")
    with ProcessPoolExecutor(max_workers=nucleos) as executor:
        futuros = {
            executor.submit(calcular_estatisticas, nome): nome for nome in nomes_bases
        }

        for futuro in tqdm(as_completed(futuros), total=len(futuros)):
            resultado = futuro.result()
            if isinstance(resultado, dict):
                soma_total += resultado["somas"]
                soma_sq_total += resultado["somas_sq"]
                contagem_total += resultado["contagem"]

    # Consolida as estatísticas
    media_global = soma_total / contagem_total
    variancia_global = (soma_sq_total / contagem_total) - (media_global**2)
    std_global = np.sqrt(np.maximum(variancia_global, 1e-8))

    np.save(
        "dataset/estatisticas_globais.npy", {"media": media_global, "std": std_global}
    )
    print("\n[INFO] Estatísticas Salvas!")

    print(f"\n--- PASSADA 2: Gerando Dataset Final Padronizado (Salvando no Disco) ---")
    with ProcessPoolExecutor(max_workers=nucleos) as executor:
        futuros2 = {
            executor.submit(gerar_dataset_final, nome, media_global, std_global): nome
            for nome in nomes_bases
        }

        for futuro in tqdm(as_completed(futuros2), total=len(futuros2)):
            resultado = futuro.result()
            if isinstance(resultado, str):
                print(resultado)

    print("\nProcessamento direto e sem duplicidade de disco concluído!")
