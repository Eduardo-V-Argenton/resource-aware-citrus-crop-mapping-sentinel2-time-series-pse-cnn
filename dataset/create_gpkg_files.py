#https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2024/UFs/SP/

import os
import warnings
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import shapes

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGs
# =====================================================================
base_dir = 'dataset/TIF/'
output_dir = 'dataset/GPKG/'
sp_shp = 'dataset/SP_RGI_2024/SP_RG_Imediatas_2024.shp'

os.makedirs(output_dir, exist_ok=True)
min_area_ha = 1.0

#Mapbiomas dict
classes_remove = [0,22,23,24,30,75,25,26,33,31,27]
rgi = [
    'São João da Boa Vista', 'Catanduva', 'Ourinhos'
]

# =====================================================================
# MASK
# =====================================================================
print("-> Preparing target cities masks")
gdf_mun = gpd.read_file(sp_shp)
gdf_mun_filtered = gdf_mun[gdf_mun['NM_RGI'].isin(rgi)].copy()

# =====================================================================
# CROP RASTER -> VETORIZE -> FILTER
# =====================================================================
files = [f for f in os.listdir(base_dir) if f.endswith('.tif')]

for file in files:
    print(f"\n{'='*70} Processing: {file}")
    
    try:
        with rasterio.open(f"{base_dir}{file}") as src:
            # Ensure CRS
            gdf_mun_proj = gdf_mun_filtered.to_crs(src.crs)
            target_geometry = gdf_mun_proj.geometry.tolist()
            
            # Clip raster to target geometry (crop reduces memory)
            print(" -> Clipping raster by city mask")
            out_image, out_transform = mask(src, target_geometry, crop=True)
            clipped_image = out_image[0]
            
            # Create a mask with only the target classes
            print(" -> Reclassifying and cleaning noises")
            valid_mask = ~np.isin(clipped_image, classes_remove)
            
            #  Vectorize polygons
            print(" -> Vectorizing")
            polygon_generator = shapes(
                clipped_image.astype('uint8'), 
                mask=valid_mask,
                transform=out_transform
            )
            
            features = []
            for geom, pixel_value in polygon_generator:
                mapbiomas_class = int(pixel_value)
                features.append({
                    'geometry': geom, 
                    'properties': {'mapbiomas_class': mapbiomas_class}
                })
                
            gdf_raw = gpd.GeoDataFrame.from_features(features, crs=src.crs)
            
            #Filter the area (EPSG:6933)
            print(f" -> Calculate filtered area (> {min_area_ha} ha)")
            gdf_calc = gdf_raw.to_crs(epsg=6933)
            gdf_calc['area_ha'] = gdf_calc.geometry.area / 10000.0
            gdf_valid = gdf_calc[gdf_calc['area_ha'] >= min_area_ha].copy()
            
            if gdf_valid.empty:
                print(f" -> No polygon is bigger than {min_area_ha} ha.")
                continue
            
            print(" -> Simplifying geometries")
            gdf_valid['geometry'] = gdf_valid.geometry.simplify(tolerance=10.0, preserve_topology=True)
            
            print(" -> Reprojecting and inserting NM_RGI via Spatial Join")
            
            gdf_export = gdf_valid.to_crs(epsg=4326)
            gdf_mun_4326 = gdf_mun_filtered.to_crs(epsg=4326)
            
            gdf_export = gpd.sjoin(gdf_export, gdf_mun_4326[['NM_RGI', 'geometry']], how='left', predicate='intersects')
            
            final_cols = ['mapbiomas_class', 'area_ha', 'NM_RGI', 'geometry']
            
            output_name = file.replace('.tif', '.gpkg')
            gdf_export[final_cols].to_file(os.path.join(output_dir, output_name), driver="GPKG")
            print(f" -> SUCCESS: {output_name}")
            
    except Exception as e:
        print(f"Erro ao processar {file}: {e}")