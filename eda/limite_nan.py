import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

CSV_FILE = 'dataset/dataset_index.csv'
SOURCE_FOLDER = '/mnt/SSD_SATA/Tensores_Treino/'

# =====================================================================
# ANALYSIS FUNCTION (Counts clouds without saving anything)
# =====================================================================
def analyze_polygon(base_name):
    try:
        # 1. Load the tensors
        path_10m = os.path.join(SOURCE_FOLDER, f"{base_name}_10m.npy")
        path_20m = os.path.join(SOURCE_FOLDER, f"{base_name}_20m.npy")
        
        if not os.path.exists(path_10m) or not os.path.exists(path_20m):
            return None

        t_10 = np.load(path_10m).astype(np.float32) / 10000.0
        t_20 = np.load(path_20m).astype(np.float32) / 10000.0
        
        _, _, h_10, w_10 = t_10.shape
        if h_10 < 15 or w_10 < 15:
            return None
            
        # 2. ZOOM
        tensor_20_tmp = torch.from_numpy(t_20)
        t_20_zoom = F.interpolate(tensor_20_tmp, size=(h_10, w_10), mode='bilinear', align_corners=False).numpy()
        
        # 3. Extract Bands for the Mask
        b2 = t_10[:, 0, :, :]
        b3 = t_10[:, 1, :, :]
        b4 = t_10[:, 2, :, :]
        b8 = t_10[:, 3, :, :]
        
        # 4. Apply exactly the same rules as your main script
        nodata_mask = (b2 + b3 + b4 + b8) == 0
        cloud_mask = b2 > 0.15
        shadow_mask = b8 < 0.08
        destruction_mask = nodata_mask | cloud_mask | shadow_mask
        
        # Instead of calculating heavy indices, we use an array of ones to test NaNs
        test_tensor = np.ones_like(b2)
        test_tensor[destruction_mask] = np.nan
        
        # 5. Calculation of statistics for this polygon
        total_pixels = test_tensor.size
        nans_count = np.isnan(test_tensor).sum()
        nans_rate = nans_count / total_pixels
        
        # Checks if any pixel is entirely NaN across all time steps (dead pixel)
        dead_pixel = bool(np.isnan(test_tensor).all(axis=0).any())
        
        return {
            'base_name': base_name, 
            'nans_rate': nans_rate, 
            'dead_pixel': dead_pixel
        }
        
    except Exception:
        return None

# =====================================================================
# MAIN ENGINE AND REPORT
# =====================================================================
if __name__ == '__main__':
    df_csv = pd.read_csv(CSV_FILE)
    df_csv = df_csv[df_csv['ano'] >= 2019]
    base_names = df_csv['nome_base'].tolist()

    results = []
    cores = max(1, os.cpu_count() - 1)

    print(f"Starting cloud analysis on {len(base_names)} polygons...")
    
    with ProcessPoolExecutor(max_workers=cores) as executor:
        futures = {executor.submit(analyze_polygon, name): name for name in base_names}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning NaNs"):
            res = future.result()
            if res is not None:
                results.append(res)
                
    # Create a DataFrame with the results to easily calculate statistics
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print(" ATMOSPHERIC DESTRUCTION REPORT (NaNs)")
    print("="*60)
    
    total_valid = len(df_res)
    dead_count = df_res['dead_pixel'].sum()
    print(f"Total polygons successfully read: {total_valid}")
    print(f"Polygons discarded for having '100% Blind Pixels': {dead_count} ({(dead_count/total_valid)*100:.1f}%)")
    
    print("\n--- Actual Distribution of NaN Rate ---")
    rates = df_res['nans_rate']
    print(f"General Cloud Mean: {rates.mean():.1%}")
    print(f"Median (50% of the dataset has LESS than): {rates.median():.1%}")
    print(f"75th Percentile (75% of the dataset has LESS than): {rates.quantile(0.75):.1%}")
    print(f"85th Percentile (85% of the dataset has LESS than): {rates.quantile(0.85):.1%}")
    print(f"95th Percentile (95% of the dataset has LESS than): {rates.quantile(0.95):.1%}")
    
    print("\n--- LIMIT SIMULATION FOR YOUR MODEL ---")
    print("How many farms remain if you set MAX_NANS_LIMIT to:")
    
    limit_tests = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    for limit in limit_tests:
        # Approved if below the limit AND has no blind pixels
        approved = len(df_res[(df_res['nans_rate'] <= limit) & (~df_res['dead_pixel'])])
        lost = total_valid - approved
        pct_lost = (lost/total_valid)*100
        
        print(f" > Limit {limit:.2f} ({int(limit*100)}%): {approved} farms remain | Loses {lost} ({pct_lost:.1f}%)")
        
    print("="*60)