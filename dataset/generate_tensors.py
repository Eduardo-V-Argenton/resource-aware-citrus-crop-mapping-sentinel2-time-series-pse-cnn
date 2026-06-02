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
SOURCE_FOLDER = "/mnt/ssd_sata/dataset/s2_npz" 
DESTINATION_FOLDER = "dataset/Tensors/"
SKIPPED_LIST = "dataset/skipped.txt"

os.makedirs(DESTINATION_FOLDER, exist_ok=True)

num_npz = 0
skipped_npz = 0
def load_and_clean_tensor(base_name):
    try:
        with np.load(os.path.join(SOURCE_FOLDER, f"{base_name}_10m.npz")) as data:
            t_10 = data['tensor']
        with np.load(os.path.join(SOURCE_FOLDER, f"{base_name}_20m.npz")) as data:
            t_20 = data['tensor']
    except Exception as e:
        return None

    _, _, h_10, w_10 = t_10.shape

    # 20m bands upsampling (Nearest Neighbor) - Seguro para inteiros
    t_20_tensor = torch.from_numpy(t_20.astype(np.float32))
    t_20_up = F.interpolate(t_20_tensor, size=(h_10, w_10), mode="nearest").numpy().astype(np.int16)

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
    # Shape: (Time, 10 Canais, Altura, Largura)
    base_tensor = np.stack((b2, b3, b4, b8, b5, b6, b7, b8a, b11, b12), axis=1)

    # ---------------------------------------------------------
    # Polygon Masking
    # ---------------------------------------------------------
    # Encontra pixels válidos (maiores que 0 no primeiro instante de tempo da banda B2)
    farm_mask = base_tensor[0, 0, :, :] > 0 
    
    # Achata a dimensão espacial: (Time, Channels, ValidPixels)
    flat_base = base_tensor[:, :, farm_mask]
    
    t, c, p = flat_base.shape
    
    # Se o polígono for muito pequeno (menos de 32 pixels válidos de 10x10m), descarta
    if p < 32:
        return None
        
    # ---------------------------------------------------------
    # Temporal Gap-Filling (Apenas FORWARD - Sem vazamento de dados)
    # ---------------------------------------------------------
    # Como mantivemos int16, pixels de nuvem/sem dados são 0.
    nans_mask = (flat_base == 0)
    
    # Forward gap-filling: propaga a última leitura válida para os buracos futuros
    for time_step in range(1, t):
        flat_base[time_step] = np.where(
            nans_mask[time_step], 
            flat_base[time_step - 1], 
            flat_base[time_step]
        )

    return flat_base


def generate_final_dataset(base_name):
    # Prevent PyTorch from spawning excess threads inside the executor
    torch.set_num_threads(1)
    
    output_path = os.path.join(DESTINATION_FOLDER, f"{base_name}_pse.npy")

    if os.path.exists(output_path):
        return True

    try:
        tensor = load_and_clean_tensor(base_name)
        if tensor is None:
            return [2,base_name]

        np.save(output_path, tensor.astype(np.uint16))
        return [1,base_name]
        
    except Exception as e:
        return  [-1,f"Error in {base_name}: {str(e)}"]


if __name__ == "__main__":
    df_csv = pd.read_csv(CSV_FILE)

    # Ensure uniqueness if there are duplicate names
    base_names = df_csv["name"].unique().tolist()
    cores = max(1, os.cpu_count() - 2)

    print(f"\n--- Generating Final Dataset ({len(base_names)} files) ---")
    
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {
            executor.submit(generate_final_dataset, name): name
            for name in base_names
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result[0] == -1:
                print(result)
            elif result[0] == 2:
                with open(SKIPPED_LIST, "a", encoding="utf-8") as f:
                    f.write(f'{result[1]}' + "\n")

    print("\nDirect processing completed! Files ready for PyTorch.")
