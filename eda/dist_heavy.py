import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", r"Mean of empty slice")

# =====================================================================
# CONFIGURAÇÕES DE DIRETÓRIO E PARÂMETROS
# =====================================================================
FICHEIRO_CSV = '/mnt/SSD_SATA/dataset/dataset_index.csv'
PASTA_ORIGEM = "/mnt/SSD_SATA/dataset/Tensores_Treino/"

# =====================================================================
# FUNÇÃO AUXILIAR: PROCESSAMENTO BASE
# =====================================================================
def verify_small(nome_base):
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

    # Extração e Empilhamento
    b2, b3, b4, b8 = t_10[:, 0], t_10[:, 1], t_10[:, 2], t_10[:, 3]
    tensor_base = np.stack((b2, b3, b4, b8), axis=1)

    # Limpeza e Clip (-2.0 a 2.0)
    tensor_base = np.nan_to_num(tensor_base, nan=np.nan, posinf=2.0, neginf=-2.0)
    tensor_base = np.clip(tensor_base, a_min=-2.0, a_max=2.0)
    
    mascara_fazenda = (tensor_base[0, 0, :, :] != 0.0) & (tensor_base[0, 0, :, :] > -0.5)
    flat_base = tensor_base[:, :, mascara_fazenda]
    t, c, p = flat_base.shape
    if p < 32:
        return False
    return True

# =====================================================================
# PROCESSAMENTO E SALVAMENTO (Sem Normalização)
# =====================================================================

data = pd.read_csv('/mnt/SSD_SATA/dataset/dataset_index.csv')
len_data = len(data)
counter_small = 0
for i,d in enumerate(data.itertuples(index=False)):
    print(f"{i}/{len_data}", end="\r")
    if not verify_small(d.name):
        data.drop(data[data['name'] == d.name].index, inplace=True)
        counter_small += 1

len_total = len(data)
len_citrus = len(data[data["mapbiomas_class"] == 47])
len_cana = len(data[data["mapbiomas_class"] == 20])
len_cafe = len(data[data["mapbiomas_class"] == 46])
len_pastagem = len(data[data["mapbiomas_class"] == 15])
len_floresta = len(data[data["mapbiomas_class"] == 3])
len_soja = len(data[data["mapbiomas_class"] == 39])
len_sivicultura = len(data[data["mapbiomas_class"] == 9])
len_alagado = len(data[data["mapbiomas_class"] == 11])
len_false =  len_cana + len_cafe + len_pastagem + len_floresta + len_soja + len_sivicultura + len_alagado

print(f"Total de amostras: {len_total}")
print(f"Total de amostras de citros: {len_citrus} ({len_citrus/len_total:.2%})")
print(f"Total de amostras de cana: {len_cana} ({len_cana/len_total:.2%})")
print(f"Total de amostras de café: {len_cafe} ({len_cafe/len_total:.2%})")
print(f"Total de amostras de pastagem: {len_pastagem} ({len_pastagem/len_total:.2%})")
print(f"Total de amostras de floresta: {len_floresta} ({len_floresta/len_total:.2%})")
print(f"Total de amostras de soja: {len_soja} ({len_soja/len_total:.2%})")
print(f"Total de amostras de silvicultura: {len_sivicultura} ({len_sivicultura/len_total:.2%})")
print(f"Total de amostras de campo alagado: {len_alagado} ({len_alagado/len_total:.2%})")
print(f"Total de amostras false: {len_false/len_total:.2%}")
print(f"Total de amostras removidas por serem pequenas: {counter_small} ({counter_small/len_data:.2%})")
