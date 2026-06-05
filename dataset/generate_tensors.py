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
# DIRECTORY CONFIGURATIONS AND PARAMETERS
# =====================================================================
CSV_FILE = '/mnt/ssd_sata/dataset/dataset_index.csv'
SOURCE_FOLDER = "/home/eduardo/Desktop/dataset/Tensors/"
DESTINATION_FOLDER = "dataset/Tensors/"

QUALITY_LOG = "dataset/quality_report.csv" 

os.makedirs(DESTINATION_FOLDER, exist_ok=True)


def load_and_clean_tensor(base_name):
    """Loads bands, performs upsampling, clipping, and gap-filling. Returns tensor with NaNs in the background."""
    t_10 = (
        np.load(os.path.join(SOURCE_FOLDER, f"{base_name}_10m.npy")).astype(np.float32)
        / 10000.0
    )
    t_20 = (
        np.load(os.path.join(SOURCE_FOLDER, f"{base_name}_20m.npy")).astype(np.float32)
        / 10000.0
    )

    _, _, h_10, w_10 = t_10.shape

    # 20m bands upsampling (Nearest Neighbor)
    t_20_tensor = torch.from_numpy(t_20)
    t_20_up = F.interpolate(t_20_tensor, size=(h_10, w_10), mode="nearest").numpy()

    # Extraction and Stacking
    b2, b3, b4, b8 = t_10[:, 0], t_10[:, 1], t_10[:, 2], t_10[:, 3]
    b5, b6, b7, b8a, b11, b12 = (
        t_20_up[:, 0],
        t_20_up[:, 1],
        t_20_up[:, 2],
        t_20_up[:, 3],
        t_20_up[:, 4],
        t_20_up[:, 5],
    )
    base_tensor = np.stack((b2, b3, b4, b8, b5, b6, b7, b8a, b11, b12), axis=1)

    # Cleaning and Clipping (-2.0 to 2.0)
    base_tensor = np.nan_to_num(base_tensor, nan=np.nan, posinf=2.0, neginf=-2.0)
    base_tensor = np.clip(base_tensor, a_min=-2.0, a_max=2.0)
    
    farm_mask = (base_tensor[0, 0, :, :] != 0.0) & (base_tensor[0, 0, :, :] > -0.5)
    flat_base = base_tensor[:, :, farm_mask]
    t, c, p = flat_base.shape
    if p < 32:
        return None, None
        
    # =================================================================
    # CÁLCULO DE DADOS ARTIFICIAIS ADICIONADO AQUI
    # =================================================================
    nans_mask = np.isnan(flat_base) | (flat_base == 0.0)
    
    total_valores = flat_base.size
    valores_artificiais = np.sum(nans_mask)
    perc_artificial = (valores_artificiais / total_valores) * 100.0
    
    # Forward gap-filling
    for time_step in range(1, t):
        flat_base[time_step] = np.where(
            nans_mask[time_step], flat_base[time_step - 1], flat_base[time_step]
        )

    # Backward gap-filling
    # nans_mask_bwd = np.isnan(flat_base) | (flat_base == 0.0)
    # for time_step in range(t - 2, -1, -1):
    #     flat_base[time_step] = np.where(
    #         nans_mask_bwd[time_step], flat_base[time_step + 1], flat_base[time_step]
    #     )

    # Agora retorna também a porcentagem
    return flat_base, perc_artificial


def generate_final_dataset(base_name):
    torch.set_num_threads(1)
    output_path = os.path.join(DESTINATION_FOLDER, f"{base_name}_pse.npy")

    if os.path.exists(output_path):
        return [0, base_name, 0.0]

    try:
        tensor, perc_artificial = load_and_clean_tensor(base_name)
        if tensor is None:
            return [2, base_name, 0.0]

        final_tensor = np.nan_to_num(tensor, nan=0.0).astype(np.float16)

        np.save(output_path, final_tensor)
        return [1, base_name, perc_artificial]
        
    except Exception as e:
        return [-1, f"Error in {base_name}: {str(e)}", 0.0]


if __name__ == "__main__":
    df_csv = pd.read_csv(CSV_FILE)

    base_names = df_csv["name"].tolist()
    cores = max(1, os.cpu_count() - 2)

    print("\n--- Generating Final Dataset (Cleaning and Gap Filling) ---")
    
    # Inicializa o arquivo de relatório de qualidade com cabeçalho
    with open(QUALITY_LOG, "w", encoding="utf-8") as f:
        f.write("base_name,perc_artificial\n")
    
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {
            executor.submit(generate_final_dataset, name): name
            for name in base_names
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            status, info, perc = future.result()
            
            if status == -1:
                # Ocorreu um erro
                print(info)
            elif status == 1:
                # Sucesso: Salva a porcentagem no arquivo de log
                with open(QUALITY_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{info},{perc:.2f}\n")

    print("\nDirect processing completed! Files ready for PyTorch.")
    print(f"Log de qualidade salvo em: {QUALITY_LOG}")
