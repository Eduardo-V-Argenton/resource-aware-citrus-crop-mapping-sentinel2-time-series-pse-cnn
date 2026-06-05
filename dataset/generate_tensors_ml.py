import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore", r"Mean of empty slice")

# =====================================================================
CSV_FILE = '/mnt/ssd_sata/dataset/dataset_index.csv'
SOURCE_FOLDER = "dataset/Tensors/" 
OUTPUT = "dataset/dataset_ml.csv"

BAND_NAMES = ['B2', 'B3', 'B4', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']

def extract_tabular_features(base_name):
    torch.set_num_threads(1)
    
    tensor_path = os.path.join(SOURCE_FOLDER, f"{base_name}_pse.npy")
    
    if not os.path.exists(tensor_path):
        return None 

    try:
        tensor = np.load(tensor_path).astype(np.float32)
        
        farm_mean_series = np.nanmean(tensor, axis=2) 
        
        features = {'image_name': base_name}
        
        for c, band_name in enumerate(BAND_NAMES):
            band_series = farm_mean_series[:, c]
            
            valid_mask = ~np.isnan(band_series)
            valid_series = band_series[valid_mask]
            
            if len(valid_series) == 0:
                features.update({
                    f'{band_name}_max': 0.0, f'{band_name}_min': 0.0,
                    f'{band_name}_median': 0.0, f'{band_name}_mean': 0.0,
                    f'{band_name}_std': 0.0, f'{band_name}_slope_up': 0.0,
                    f'{band_name}_slope_down': 0.0, f'{band_name}_amplitude': 0.0,
                    f'{band_name}_argmax': 0
                })
                continue
                
            features[f'{band_name}_max'] = np.max(valid_series)
            features[f'{band_name}_min'] = np.min(valid_series)
            features[f'{band_name}_median'] = np.median(valid_series)
            features[f'{band_name}_mean'] = np.mean(valid_series)
            features[f'{band_name}_std'] = np.std(valid_series)
            
            diffs = np.diff(valid_series)
            features[f'{band_name}_slope_up'] = np.sum(diffs[diffs > 0])
            features[f'{band_name}_slope_down'] = np.sum(np.abs(diffs[diffs < 0]))
            
            features[f'{band_name}_amplitude'] = np.max(valid_series) - np.min(valid_series)
            features[f'{band_name}_argmax'] = np.argmax(valid_series)
            
        return features
        
    except Exception as e:
        print(f"Erro em {base_name}: {str(e)}")
        return None


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df_csv = pd.read_csv(CSV_FILE)

    base_names = df_csv["name"].tolist()
    cores = max(1, os.cpu_count() - 2)

    print("\n--- Extracting Tabular Features for Machine Learning ---")
    
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
    
    # Cria o DataFrame final e salva
    df_ml = pd.DataFrame(tabular_data)
    
    # Reordenando para garantir que o image_name fique na primeira coluna
    cols = ['image_name'] + [c for c in df_ml.columns if c != 'image_name']
    df_ml = df_ml[cols]
    
    df_ml.to_csv(OUTPUT, index=False)
    
    print(f"File successfully saved at: {OUTPUT}")
    print(f"Total Rows: {len(df_ml)} | Total Columns: {len(df_ml.columns)}")
