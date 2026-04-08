import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore", r"Mean of empty slice")
warnings.filterwarnings("ignore", message=".*All-NaN slice encountered.*")

# =====================================================================
# CONFIGURAÇÕES DE DIRETÓRIO E PARÂMETROS
# =====================================================================
FICHEIRO_CSV = "dataset/dataset_index.csv"
PASTA_ORIGEM = "/mnt/SSD_SATA/Tensores_Treino/"
ARQUIVO_SAIDA_EDA = "dataset_eda_temporal_20pct.csv"

# Classes alvo do seu estudo
CLASS_MAP = {
    47: "Citrus",
    20: "Cana",
    46: "Cafe",
    15: "Pastagem",
    9: "Silvicultura",
    39: "Soja",
    3: "Floresta",
    11: "Pantano",
}

LISTA_INDICES = [
    "EVI",
    "NDRE",
    "NDTI",
    "NDMI",
    "BSI",
    "TCARI",
    "NDVI",
    "GNDVI",
    "SAVI",
    "MSI",
    "VARI",
    "MCARI",
    "CIRED_EDGE",
    "PSRI",
    "NDWI",
    "MSAVI",
    "OSAVI",
    "ARVI",
    "SIPI",
    "NDGI",
    "CI_GREEN",
]


def processar_poligono_para_eda(tupla_dados):
    # Previne que o PyTorch tente multithreading dentro do multiprocessing (Deadlock)
    torch.set_num_threads(1)

    nome_base, classe_id = tupla_dados

    try:
        # 1. Carregamento dos dados brutos
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

        if h_10 < 5 or w_10 < 5:
            return None  # Ignora polígonos literalmente minúsculos ou inválidos

        # 2. Interpolação Bilinear (Ajustando a resolução de 20m para 10m)
        tensor_20_tmp = torch.from_numpy(t_20)
        t_20_zoom = F.interpolate(
            tensor_20_tmp, size=(h_10, w_10), mode="bilinear", align_corners=False
        ).numpy()

        # 3. Separação das Bandas (USANDO A IMAGEM COMPLETA)
        b2 = t_10[:, 0]  # BLUE
        b3 = t_10[:, 1]  # GREEN
        b4 = t_10[:, 2]  # RED
        b8 = t_10[:, 3]  # NIR
        b5 = t_20_zoom[:, 0]  # RE1
        b6 = t_20_zoom[:, 1]  # RE2
        b11 = t_20_zoom[:, 4]  # SWIR1
        b12 = t_20_zoom[:, 5]  # SWIR2

        # 4. Cálculo dos Índices Pixel a Pixel (Tratando divisões por zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            indices = {
                "EVI": np.where(
                    (b8 + 6 * b4 - 7.5 * b2 + 1) != 0,
                    2.5 * ((b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1)),
                    np.nan,
                ),
                "NDRE": np.where((b8 + b5) != 0, (b8 - b5) / (b8 + b5), np.nan),
                "NDTI": np.where((b11 + b12) != 0, (b11 - b12) / (b11 + b12), np.nan),
                "NDMI": np.where((b8 + b11) != 0, (b8 - b11) / (b8 + b11), np.nan),
                "BSI": np.where(
                    ((b11 + b4) + (b8 + b2)) != 0,
                    ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2)),
                    np.nan,
                ),
                "TCARI": np.where(
                    b4 != 0, 3 * ((b5 - b4) - 0.2 * (b5 - b3) * (b5 / b4)), np.nan
                ),
                "NDVI": np.where((b8 + b4) != 0, (b8 - b4) / (b8 + b4), np.nan),
                "GNDVI": np.where((b8 + b3) != 0, (b8 - b3) / (b8 + b3), np.nan),
                "SAVI": np.where(
                    (b8 + b4 + 0.5) != 0, 1.5 * ((b8 - b4) / (b8 + b4 + 0.5)), np.nan
                ),
                "MSI": np.where(b8 != 0, b11 / b8, np.nan),
                "VARI": np.where(
                    (b3 + b4 - b2) != 0, (b3 - b4) / (b3 + b4 - b2), np.nan
                ),
                "MCARI": np.where(
                    b4 != 0, ((b5 - b4) - 0.2 * (b5 - b3)) * (b5 / b4), np.nan
                ),
                "CIRED_EDGE": np.where(b5 != 0, (b8 / b5) - 1.0, np.nan),
                "PSRI": np.where(b6 != 0, (b4 - b2) / b6, np.nan),
                "NDWI": np.where((b8 + b11) != 0, (b8 - b11) / (b8 + b11), np.nan),
                "MSAVI": (
                    2 * b8
                    + 1
                    - np.sqrt(np.clip((2 * b8 + 1) ** 2 - 8 * (b8 - b4), 0, None))
                )
                / 2,
                "OSAVI": np.where(
                    (b8 + b4 + 0.16) != 0, (b8 - b4) / (b8 + b4 + 0.16), np.nan
                ),
                "ARVI": np.where(
                    (b8 + (2 * b4 - b2)) != 0,
                    (b8 - (2 * b4 - b2)) / (b8 + (2 * b4 - b2)),
                    np.nan,
                ),
                "SIPI": np.where((b8 - b4) != 0, (b8 - b2) / (b8 - b4), np.nan),
                "NDGI": np.where((b3 + b4) != 0, (b3 - b4) / (b3 + b4), np.nan),
                "CI_GREEN": np.where(b3 != 0, (b8 / b3) - 1, np.nan),
            }

        # 5. Agregação Espacial e Construção da Linha da Tabela
        linha_resultado = {
            "nome_base": nome_base,
            "classe_id": classe_id,
            "cultura": CLASS_MAP.get(classe_id, "Outros"),
            "label_ia": 1 if classe_id == 47 else 0,
        }

        # Tira a média espacial de toda a extensão do polígono
        for idx_name in LISTA_INDICES:
            matriz_indice = indices[idx_name]  # Formato: (25 dias, Y, X)

            # Limpa anomalias matemáticas extremas antes da média
            matriz_indice = np.clip(
                np.nan_to_num(matriz_indice, nan=np.nan, posinf=5.0, neginf=-5.0),
                -5.0,
                5.0,
            )

            # Calcula a média da imagem completa ignorando NaNs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                medias_temporais = np.nanmean(matriz_indice, axis=(1, 2))

            # Interpolação temporal linear (preenche as falhas das nuvens)
            medias_temporais = (
                pd.Series(medias_temporais)
                .interpolate(limit_direction="both")
                .to_numpy()
            )
            medias_temporais = np.nan_to_num(medias_temporais, nan=0.0)

            for dia in range(25):
                linha_resultado[f"{idx_name}_d{dia + 1}"] = round(
                    float(medias_temporais[dia]), 4
                )

        return linha_resultado

    except Exception as e:
        return None


if __name__ == "__main__":
    print("Lendo CSV mestre...")
    df_csv = pd.read_csv(FICHEIRO_CSV)
    df_csv = df_csv[df_csv["ano"] >= 2019]

    # Filtra apenas as classes que importam para o estudo
    df_csv = df_csv[df_csv["classe_mapbiomas"].isin(CLASS_MAP.keys())]
    print(f"Total de polígonos válidos: {len(df_csv)}")

    # =========================================================
    # AMOSTRAGEM ESTRATIFICADA (20%)
    # Garante proporções perfeitas de anos e de culturas (label)
    # =========================================================
    print("Aplicando amostragem estratificada (20%)...")
    df_csv = df_csv.groupby(["ano", "classe_mapbiomas"], group_keys=False).sample(
        frac=0.2, random_state=42
    )
    print(f"Total de polígonos após amostragem: {len(df_csv)}")

    # Monta a lista de tarefas (nome_base, classe)
    tarefas = list(zip(df_csv["nome_base"], df_csv["classe_mapbiomas"]))

    print(
        f"\nIniciando extração tabular de imagem completa para {len(tarefas)} polígonos..."
    )
    resultados = []

    # Usa núcleos da CPU de forma segura
    nucleos = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=nucleos) as executor:
        futuros = {
            executor.submit(processar_poligono_para_eda, tarefa): tarefa
            for tarefa in tarefas
        }

        for futuro in tqdm(
            as_completed(futuros), total=len(futuros), desc="Gerando CSV de EDA"
        ):
            res = futuro.result()
            if res is not None:
                resultados.append(res)

    if resultados:
        df_eda = pd.DataFrame(resultados)
        df_eda.to_csv(ARQUIVO_SAIDA_EDA, index=False)
        print(f"\n[OK] Extração concluída! Arquivo salvo como: {ARQUIVO_SAIDA_EDA}")
        print(f"Shape do Dataset Final: {df_eda.shape} (Polígonos x Features)")
    else:
        print(
            "\nErro: Nenhum polígono foi processado com sucesso. Verifique os dados de origem."
        )
