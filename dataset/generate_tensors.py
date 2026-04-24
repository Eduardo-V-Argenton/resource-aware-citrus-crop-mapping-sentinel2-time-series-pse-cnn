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
CSV_FILE = '/mnt/SSD_SATA/dataset/dataset_index.csv'
SOURCE_FOLDER = "/mnt/SSD_SATA/dataset/Tensores_Treino/"
DESTINATION_FOLDER = "dataset/Tensors/"

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
        return None
        
    # Forward gap-filling
    nans_mask = np.isnan(flat_base) | (flat_base == 0.0)
    for time_step in range(1, t):
        flat_base[time_step] = np.where(
            nans_mask[time_step], flat_base[time_step - 1], flat_base[time_step]
        )

    # Backward gap-filling
    nans_mask_bwd = np.isnan(flat_base) | (flat_base == 0.0)
    for time_step in range(t - 2, -1, -1):
        flat_base[time_step] = np.where(
            nans_mask_bwd[time_step], flat_base[time_step + 1], flat_base[time_step]
        )

    return flat_base


def generate_final_dataset(base_name):
    torch.set_num_threads(1)
    output_path = os.path.join(DESTINATION_FOLDER, f"{base_name}_pse.npy")

    if os.path.exists(output_path):
        return True

    try:
        tensor = load_and_clean_tensor(base_name)
        if tensor is None:
            return None

        final_tensor = np.nan_to_num(tensor, nan=0.0).astype(np.float16)

        np.save(output_path, final_tensor)
        return True
    except Exception as e:
        return f"Error in {base_name}: {str(e)}"


if __name__ == "__main__":
    df_csv = pd.read_csv(CSV_FILE)

    base_names = df_csv["name"].tolist()
    cores = max(1, os.cpu_count() - 2)

    print("\n--- Generating Final Dataset (Cleaning and Gap Filling) ---")
    
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {
            executor.submit(generate_final_dataset, name): name
            for name in base_names
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if isinstance(result, str):
                print(result)

    print("\nDirect processing completed! Files ready for PyTorch.")
