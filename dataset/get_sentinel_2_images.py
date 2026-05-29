import os
import time
import warnings
import requests
import random
import threading
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.io import MemoryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import box, mapping
import ee
import pyproj
from shapely.ops import transform

warnings.filterwarnings("ignore")

# =====================================================================
# Configs
# =====================================================================
PROJECT_GEE = 'clean-bindery-462116-u8' 

print("-> Connecting to Google Earth Engine...")
try:
    ee.Initialize(project=PROJECT_GEE, opt_url='https://earthengine-highvolume.googleapis.com')
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_GEE, opt_url='https://earthengine-highvolume.googleapis.com')

output_folder = 'dataset/dataset'
csv_output = 'dataset/dataset_index.csv'
os.makedirs(output_folder, exist_ok=True)
csv_lock = threading.Lock()

error_lock = threading.Lock()
consecutive_errors = 0
MAX_CONSECUTIVE_ERRORS = 15 
circuit_breaker = threading.Event()
circuit_breaker.set() 

# =====================================================================
# SELF-HEALING
# =====================================================================
print("\n" + "="*60 + "\n Auditing and Healing\n" + "="*60)
processed = set()

if os.path.exists(csv_output):
    try:
        df_audit = pd.read_csv(csv_output)
        
        def validate_tensors(nome):
            c10, c20 = os.path.join(output_folder, f"{nome}_10m.npy"), os.path.join(output_folder, f"{nome}_20m.npy")
            return os.path.exists(c10) and os.path.getsize(c10) > 1000 and os.path.exists(c20) and os.path.getsize(c20) > 1000

        print(" -> Verifying tensor integrity for entries in CSV...")
        masks_valid = df_audit['name'].apply(validate_tensors)
        df_cleaned = df_audit[masks_valid].copy()
        processed.update(df_cleaned['name'].tolist())
        
        df_cleaned.to_csv(csv_output, index=False)
        print(f" -> Audit CSV: {len(df_audit)} originals -> {len(df_cleaned)} health.")
        
        print(" -> Removing orphans")
        on_disk_files = os.listdir(output_folder)
        removed = 0
        for f in on_disk_files:
            if f.endswith('.npy'):
                base_file_name = f.replace('_10m.npy', '').replace('_20m.npy', '')
                if base_file_name not in processed:
                    os.remove(os.path.join(output_folder, f))
                    removed += 1
        print(f" -> Cleaning completed: {removed} files deleted.")
    except Exception as e:
        print(f" -> Error: {e}.")
else:
    print(" -> CSV not found. Starting new dataset")

transformer_to_m = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
transformer_to_deg = pyproj.Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)

# =====================================================================
# Download Data
# =====================================================================
def download_gee(geom_geojson, year, part_name, max_tries=3):
    geom = ee.Geometry(geom_geojson)
    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='15D')
    
    NODATA_VALUE = -9999
    base_10m = ee.Image.constant([NODATA_VALUE]*4).rename(['B2', 'B3', 'B4', 'B8']).toInt16()
    base_20m = ee.Image.constant([NODATA_VALUE]*6).rename(['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']).toInt16()
    

    list_10m, list_20m = [], []

    for i in range(len(dates)-1):
        start, end = dates[i].strftime('%Y-%m-%d'), dates[i+1].strftime('%Y-%m-%d')
        col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(geom).filterDate(start, end)
        
        def mask_scl(img):
            scl = img.select('SCL')
            
            mask = scl.neq(3) \
                .And(scl.neq(8)) \
                .And(scl.neq(9)) \
                .And(scl.neq(10)) \
                .And(scl.neq(11))
            return img.updateMask(mask)
        
        def processar(col_valida):
            median = col_valida.map(mask_scl).median()
            img_10 = median.select(['B2', 'B3', 'B4', 'B8']).unmask(0).clip(geom).unmask(NODATA_VALUE).toInt16()
            img_20 = median.select(['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']).unmask(0).clip(geom).unmask(NODATA_VALUE).toInt16()
            return ee.Dictionary({'10m': img_10, '20m': img_20})
            
        res = ee.Algorithms.If(col.size().gt(0), processar(col), ee.Dictionary({'10m': base_10m, '20m': base_20m}))
        res_dict = ee.Dictionary(res)
        
        list_10m.append(ee.Image(res_dict.get('10m')))
        list_20m.append(ee.Image(res_dict.get('20m')))

    stack_10m = ee.ImageCollection(list_10m).toBands()
    stack_20m = ee.ImageCollection(list_20m).toBands()

    for attempt in range(max_tries):
        try:
            # Baixa 10m
            url_10m = stack_10m.getDownloadURL({'region': geom, 'scale': 10, 'format': 'GEO_TIFF'})
            with MemoryFile(requests.get(url_10m, timeout=60).content) as memfile:
                with memfile.open() as src:
                    arr_10m_flat = src.read()
                    
            # Baixa 20m
            url_20m = stack_20m.getDownloadURL({'region': geom, 'scale': 20, 'format': 'GEO_TIFF'})
            with MemoryFile(requests.get(url_20m, timeout=60).content) as memfile:
                with memfile.open() as src:
                    arr_20m_flat = src.read()

            num_periodos = len(dates)-1
            _, H10, W10 = arr_10m_flat.shape
            _, H20, W20 = arr_20m_flat.shape
            
            tensor_10m = arr_10m_flat.reshape((num_periodos, 4, H10, W10))
            tensor_20m = arr_20m_flat.reshape((num_periodos, 6, H20, W20))
            
            tensor_10m[tensor_10m == NODATA_VALUE] = 0
            tensor_20m[tensor_20m == NODATA_VALUE] = 0
            
            b2_temporal = tensor_10m[:, 0, :, :] 
                        
            volume_total = b2_temporal.size 
            
            pixels_valid = np.count_nonzero(b2_temporal > 0)
            
            mask_outside_polygon = np.all(b2_temporal == 0, axis=0) 
            pixels_padding = np.count_nonzero(mask_outside_polygon) * num_periodos
            
            pixels_zeros_total = volume_total - pixels_valid
            pixels_cloud_scl = pixels_zeros_total - pixels_padding

            prop_valid = round((pixels_valid / volume_total) * 100, 2)
            prop_padding = round((pixels_padding / volume_total) * 100, 2)
            prop_scl = round((pixels_cloud_scl / volume_total) * 100, 2)
            
            metrics = {
                'prop_valid_%': prop_valid,
                'prop_padding_%': prop_padding,
                'prop_scl_%': prop_scl
            }
            return tensor_10m, tensor_20m, metrics

        except Exception as e:
            error_msg = str(e).lower()
                        
            if 'user memory limit' in error_msg or 'payload too large' in error_msg or 'computed value is too large' in error_msg:
                print(f"      [FATAL] GEE recusou {part_name} (Muito pesado/Complexo). Pulando...")
                return None, None, {}
            
            if 'quota' in error_msg or 'too many concurrent' in error_msg or '429' in error_msg:
                sleep_time = 60 + random.uniform(5, 15)
                print(f"      [QUOTA] GEE lotado. {part_name} pausando {sleep_time:.0f}s...")
            
            else:
                sleep_time = (5 ** attempt) + random.uniform(1, 5)
                print(f"      [WARNING] {part_name} falhou ({attempt+1}/{max_tries}). Retentando em {sleep_time:.0f}s...")
                
            time.sleep(sleep_time)

    return None, None, {}

# =====================================================================
# Process
# =====================================================================
def process_polygon(index, row, year):
    name = f"img_{index:05d}_{year}"
    geom_true_deg = row.geometry
    geom_proj = transform(transformer_to_m.transform, geom_true_deg)
    
    # If polygon is very large we clip it
    MAX_METERS = 1500.0
    minx, miny, maxx, maxy = geom_proj.bounds
    w = maxx - minx
    h = maxy - miny
    
    clipped_geoms = []
    
    if w <= MAX_METERS and h <= MAX_METERS:
        clipped_geoms = [(geom_true_deg, name, 0)]
    else:
        num_x = int(np.ceil(w / MAX_METERS))
        num_y = int(np.ceil(h / MAX_METERS))
        step_x = w / num_x
        step_y = h / num_y
        
        counter = 0
        for i in range(num_x):
            for j in range(num_y):
                clip = box(
                    minx + i * step_x,
                    miny + j * step_y,
                    minx + (i + 1) * step_x,
                    miny + (j + 1) * step_y
                )
                if geom_proj.intersects(clip):
                    counter += 1
                    geom_intersec_proj = geom_proj.intersection(clip)
                    geom_intersec_deg = transform(transformer_to_deg.transform, geom_intersec_proj)
                    clipped_geoms.append((geom_intersec_deg, f"img_{index:05d}_p{counter:02d}_{year}", 1))

    results = []
    
    for geom_piece, name, is_piece in clipped_geoms:
        if name in processed:
            results.append({'status': 'skipped', 'name': name})
            continue

        if not circuit_breaker.is_set():
            circuit_breaker.wait()

        
        tensor_10m, tensor_20m, metrics = download_gee(mapping(geom_piece), year, name)

        global consecutive_errors
        
        if tensor_10m is not None and tensor_10m.shape[2] > 0 and tensor_20m is not None and tensor_20m.shape[2] > 0:
            tensor_10m = tensor_10m.astype(np.int16)
            tensor_20m = tensor_20m.astype(np.int16)
            
            np.savez_compressed(os.path.join(output_folder, f"{name}_10m.npz"), tensor=tensor_10m)
            np.savez_compressed(os.path.join(output_folder, f"{name}_20m.npz"), tensor=tensor_20m)

            with error_lock:
                consecutive_errors = 0
                            
            results.append({
                'status': 'success', 
                'name': name,
                'data': {
                    'id': f"loc_{round(geom_piece.centroid.x, 4)}_{round(geom_piece.centroid.y, 4)}", 
                    'name': name, 
                    'mapbiomas_class': row['mapbiomas_class'], 
                    'year': year,
                    'area_ha': round(row['area_ha'], 2),
                    'is_piece': is_piece,
                    'total_pixels': tensor_10m.shape[2] * tensor_10m.shape[3],
                    'valids_%': metrics['prop_valid_%'],
                    'padding_box_%': metrics['prop_padding_%'],
                    'cloud_scl_%': metrics['prop_scl_%']
                }
            })
        else:
            with error_lock:
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS and circuit_breaker.is_set():
                    print(f"\n[!!!] CIRCUIT BREAKER ACIONADO! {consecutive_errors} erros seguidos.")
                    print("[!!!] A API do Earth Engine pode ter te bloqueado. Pausando TODAS as threads por 5 minutos...\n")
                    circuit_breaker.clear() 
                    
                    def reset_breaker():
                        time.sleep(300)
                        global consecutive_errors
                        consecutive_errors = 0
                        print("\n[OK] Retomando os downloads...\n")
                        circuit_breaker.set()
                        
                    threading.Thread(target=reset_breaker).start()

            results.append({'status': 'error', 'name': name})

    return results

def secure_add_csv(register, path):
    with csv_lock:
        pd.DataFrame([register]).to_csv(path, mode='a', header=not os.path.exists(path), index=False, encoding='utf-8')
        processed.add(register['name'])

# =====================================================================
# 5. LOOP DE EXECUÇÃO
# =====================================================================
for year in range(2019, 2025):
    print(f"\n{'='*50}\n Executing year: {year}")
    gpkg_file = f'dataset/GPKG/mapbiomas-brazil-collection-101-saopaulosp-{year}.gpkg' 
    
    if not os.path.exists(gpkg_file):
        continue
        
    gdf_final = gpd.read_file(gpkg_file).to_crs(epsg=4326).drop_duplicates(subset='geometry').reset_index(drop=True)
    
    success, skipped = 0, 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        tasks = [executor.submit(process_polygon, idx, row, year) for idx, row in gdf_final.iterrows()]
        
        for futuro in as_completed(tasks):
            for res in futuro.result():
                if res['status'] == 'skipped':
                    skipped += 1
                elif res['status'] == 'success':
                    success += 1
                    secure_add_csv(res['data'], csv_output)
                    print(f"    [OK] {res['name']} saved!")
                else:
                    print(f"    [ERROR] Failed for {res['name']}.")

    print(f"Resume {year}: {success} new, {skipped} already existed.")