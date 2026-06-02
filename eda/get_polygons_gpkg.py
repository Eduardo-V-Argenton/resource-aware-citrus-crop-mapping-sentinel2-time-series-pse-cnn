import os
import pandas as pd
import geopandas as gpd
import numpy as np
import pyproj
from shapely.ops import transform
from shapely.geometry import box

gpkg_dir = 'dataset/GPKG/'
MAX_METERS = 1500.0

transformer_to_m = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)

arquivos_gpkg = [f for f in os.listdir(gpkg_dir) if f.endswith('.gpkg')]

print("Calculando o volume exato do Dataset...\n")

for file in arquivos_gpkg:
    filepath = os.path.join(gpkg_dir, file)
    gdf = gpd.read_file(filepath)
    
    total_imagens_ano = 0
    
    for _, row in gdf.iterrows():
        geom_proj = transform(transformer_to_m.transform, row.geometry)
        minx, miny, maxx, maxy = geom_proj.bounds
        w = maxx - minx
        h = maxy - miny
        
        if w <= MAX_METERS and h <= MAX_METERS:
            total_imagens_ano += 1
        else:
            num_x = int(np.ceil(w / MAX_METERS))
            num_y = int(np.ceil(h / MAX_METERS))
            
            # Conta os pedaços que realmente interceptam o polígono original
            step_x = w / num_x
            step_y = h / num_y
            for i in range(num_x):
                for j in range(num_y):
                    clip = box(minx + i * step_x, miny + j * step_y, 
                               minx + (i + 1) * step_x, miny + (j + 1) * step_y)
                    if geom_proj.intersects(clip):
                        total_imagens_ano += 1
                        
    print(f"[{file[-9:-5]}] Total projetado: {total_imagens_ano} matrizes/tensores.")