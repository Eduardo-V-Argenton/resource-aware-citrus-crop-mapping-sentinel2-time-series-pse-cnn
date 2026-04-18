import os
import warnings
import numpy as np
import pandas as pd
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
sp_shp = 'dataset/SP_MUN_2024/SP_Municipios_2024.shp'

os.makedirs(output_dir, exist_ok=True)
min_area_ha = 1.0

#Mapbiomas dict
classes = {
    47: 1, # Citrus
    20: 0, 46: 0, 15: 0, 9: 0, 39: 0, 3: 0, 11: 0 # Others
}
cities = [
    'Aguaí', 'Águas da Prata', 'Casa Branca', 'Espírito Santo do Pinhal', 
    'Santa Cruz das Palmeiras', 'Santo Antônio do Jardim', 'São João da Boa Vista', 
    'Tambaú', 'Vargem Grande do Sul', 'Estiva Gerbi', 'Itapira', 'Mogi Guaçu', 'Mogi Mirim'
]

# =====================================================================
# MASK
# =====================================================================
print("-> Preparing target cities masks")
gdf_mun = gpd.read_file(sp_shp)
gdf_mun_filtered = gdf_mun[gdf_mun['NM_MUN'].isin(cities)].copy()

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
            
            # Create a mask with only the target classes mentioned
            print(" -> Reclassifying and cleaning noises")
            valid_mask = np.isin(clipped_image, list(classes.keys()))
            
            if not valid_mask.any():
                print(" -> No target classes found.")
                continue
                
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
                label_ia = classes[mapbiomas_class]
                features.append({
                    'geometry': geom, 
                    'properties': {'mapbiomas_class': mapbiomas_class, 'label': label_ia}
                })
                
            gdf_raw = gpd.GeoDataFrame.from_features(features, crs=src.crs)
            
            # 3.5. Filter the area (EPSG:6933)
            print(f" -> Calculate filtered area (> {min_area_ha} ha)")
            gdf_calc = gdf_raw.to_crs(epsg=6933)
            gdf_calc['area_ha'] = gdf_calc.geometry.area / 10000.0
            gdf_valid = gdf_calc[gdf_calc['area_ha'] >= min_area_ha].copy()
            
            if gdf_valid.empty:
                print(f" -> No polygon is bigger than {min_area_ha} ha.")
                continue

            print(" -> Associate municipalities")
            gdf_mun_6933 = gdf_mun_filtered.to_crs(epsg=6933)
            
            # Separate polygons that span multiple municipalities
            gdf_valid = gpd.overlay(gdf_valid, gdf_mun_6933[['NM_MUN', 'geometry']], how='intersection')
            
            if not gdf_valid.empty:
                gdf_valid['area_ha'] = gdf_valid.geometry.area / 10000.0
                gdf_valid = gdf_valid[gdf_valid['area_ha'] >= min_area_ha].copy()
            
            # Stratified Sampling (1:4)
            print(" -> Applying balancing (1:4)...")
            gdf_citrus = gdf_valid[gdf_valid['label'] == 1]
            gdf_others = gdf_valid[gdf_valid['label'] == 0]
            
            total_area_citrus = gdf_citrus['area_ha'].sum()
            
            target_area_others = total_area_citrus * 4
            
            if gdf_others['area_ha'].sum() > target_area_others:
                # Compute class proportions
                area_per_class = gdf_others.groupby('mapbiomas_class')['area_ha'].sum()
                total_others_area = area_per_class.sum()
                
                sampled_others_list = []
                
                for cls, cls_area in area_per_class.items():
                    target_cls_area = (cls_area / total_others_area) * target_area_others
                    cls_df = gdf_others[gdf_others['mapbiomas_class'] == cls].sample(frac=1, random_state=42)
                    cls_df['acum_area'] = cls_df['area_ha'].cumsum()
                    sampled_cls = cls_df[cls_df['acum_area'] <= target_cls_area].copy()
                    
                    if sampled_cls.empty and not cls_df.empty:
                        sampled_cls = cls_df.head(1).copy()
                        
                    sampled_others_list.append(sampled_cls.drop(columns=['acum_area']))
                                    
                gdf_others = pd.concat(sampled_others_list)
                    
            gdf_final = gpd.GeoDataFrame(pd.concat([gdf_citrus, gdf_others], ignore_index=True), crs="EPSG:6933")
            
            print(" -> Simplifying geometries")
            gdf_final['geometry'] = gdf_final.geometry.simplify(tolerance=10.0, preserve_topology=True)
            print(" -> Reprojecting and saving GPKG")
            gdf_export = gdf_final.to_crs(epsg=4326)
            
            final_cols = ['mapbiomas_class', 'label', 'area_ha', 'NM_MUN', 'geometry']
            output_name = file.replace('.tif', '.gpkg')
            gdf_export[final_cols].to_file(f"{output_dir}{output_name}", driver="GPKG")
            print(f" -> SUCCESS: {output_name}")
            
    except Exception as e:
        print(f"Erro ao processar {file}: {e}")