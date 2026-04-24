import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from generate_tensors import load_and_clean_tensor

warnings.filterwarnings("ignore", r"Mean of empty slice")

# =====================================================================
# DIRECTORY CONFIGURATIONS AND PARAMETERS
# =====================================================================
CSV_FILE = '/mnt/SSD_SATA/dataset/dataset_index.csv'
SOURCE_FOLDER = "/mnt/SSD_SATA/dataset/Tensores_Treino/"
OUTPUT = "dataset/dataset_ml_bands_indexes.csv"

BAND_NAMES = [
    'B2', 'B3', 'B4', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12',
    'NDVI', 'EVI', 'NDMI', 'NDTI', 'NDRE'
]

def extract_tabular_features(base_name):
    torch.set_num_threads(1)
    
    try:
        tensor = load_and_clean_tensor(base_name)
        if tensor is None:
            return None
        
        farm_mean_series = np.nanmean(tensor, axis=2)
        
        b2 = farm_mean_series[:, 0]  # Blue
        b4 = farm_mean_series[:, 2]  # Red
        b8 = farm_mean_series[:, 3]  # NIR
        b5 = farm_mean_series[:, 4]  # Red Edge 1
        b11 = farm_mean_series[:, 8] # SWIR 1
        b12 = farm_mean_series[:, 9] # SWIR 2
        
        eps = 1e-8
        
        ndvi = (b8 - b4) / (b8 + b4 + eps)
        evi = 2.5 * ((b8 - b4) / (b8 + 6.0 * b4 - 7.5 * b2 + 1.0 + eps))
        ndmi = (b8 - b11) / (b8 + b11 + eps)
        ndti = (b11 - b12) / (b11 + b12 + eps)
        ndre = (b8 - b5) / (b8 + b5 + eps)
        
        evi = np.clip(evi, -2.0, 2.0)
        
        indices_series = np.stack((ndvi, evi, ndmi, ndti, ndre), axis=1)
        farm_mean_series = np.concatenate((farm_mean_series, indices_series), axis=1)
        
        features = {'image_name': base_name}
        
        for c, band_name in enumerate(BAND_NAMES):
            band_series = farm_mean_series[:, c]
            
            band_series = np.nan_to_num(band_series, nan=0.0)
            
            features[f'{band_name}_max'] = np.max(band_series)
            features[f'{band_name}_min'] = np.min(band_series)
            features[f'{band_name}_median'] = np.median(band_series)
            features[f'{band_name}_mean'] = np.mean(band_series)
            features[f'{band_name}_std'] = np.std(band_series)
            
            diffs = np.diff(band_series)
            features[f'{band_name}_slope_up'] = np.sum(diffs[diffs > 0])
            features[f'{band_name}_slope_down'] = np.sum(np.abs(diffs[diffs < 0]))
            
            features[f'{band_name}_amplitude'] = np.max(band_series) - np.min(band_series)
            features[f'{band_name}_argmax'] = np.argmax(band_series)
            
        return features
        
    except Exception as e:
        print(f"Error in {base_name}: {str(e)}")
        return None


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df_csv = pd.read_csv(CSV_FILE)

    base_names = df_csv["name"].tolist()
    cores = max(1, os.cpu_count() - 2)

    print("\n--- Extracting Tabular Features (Including Spectral Indices) ---")
    
    tabular_data = []
    
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {
            executor.submit(extract_tabular_features, name): name
            for name in base_names
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if isinstance(result, dict):
                tabular_data.append(result)

    print("\nProcessing completed! Saving CSV...")
    
    df_ml = pd.DataFrame(tabular_data)
    
    cols = ['image_name'] + [c for c in df_ml.columns if c != 'image_name']
    df_ml = df_ml[cols]
    
    df_ml.to_csv(OUTPUT, index=False)
    
    print(f"File successfully saved at: {OUTPUT}")
    print(f"Total Rows: {len(df_ml)} | Total Columns: {len(df_ml.columns)}")