import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# =====================================================================
# CONFIGURAÇÕES DE DIRETÓRIOS E PARÂMETROS
# =====================================================================
INDEX_FILE = '/mnt/ssd_sata/dataset/dataset_index.csv'
PREDICTIONS_FOLDER = 'results/pse_cnn/raw_predictions/'
OUTPUT_FOLDER = 'figures/charts/'
ANO_ALVO = '2023' 

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================================================
# 1. CARREGAR E PROCESSAR O INDEX
# =====================================================================
print("A carregar o dataset index...")
df_index = pd.read_csv(INDEX_FILE)

def extrair_coordenadas(id_str):
    try:
        partes = str(id_str).split('_')
        if len(partes) >= 3:
            return float(partes[1]), float(partes[2])
    except:
        pass
    return None, None

df_index['lon'], df_index['lat'] = zip(*df_index['id'].apply(extrair_coordenadas))
df_index = df_index.dropna(subset=['lon', 'lat']).copy()
df_index['geometry'] = df_index.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

# =====================================================================
# 2. CARREGAR PREDIÇÕES E FAZER O CONSENSO (VOTAÇÃO MAIORITÁRIA)
# =====================================================================
prediction_files = glob.glob(os.path.join(PREDICTIONS_FOLDER, f'*year_{ANO_ALVO}_*.csv'))

if not prediction_files:
    print(f"ERRO: Nenhum ficheiro do ano {ANO_ALVO} encontrado em {PREDICTIONS_FOLDER}")
    exit()

print(f"Foram encontrados {len(prediction_files)} ficheiros (seeds) para o ano {ANO_ALVO}.")
print("A calcular o Consenso por Votação Maioritária (Majority Vote)...")

lista_dataframes = [pd.read_csv(f) for f in prediction_files]
df_todas_seeds = pd.concat(lista_dataframes, ignore_index=True)

df_consenso = df_todas_seeds.groupby('id_sample').agg(
    y_true=('y_true', 'first'), 
    mean_probability=('model_probability', 'mean'), 
    votos_positivos=('final_threshold_prediction', 'sum'),
    total_votos=('final_threshold_prediction', 'count') 
).reset_index()

df_consenso['final_threshold_prediction'] = (df_consenso['votos_positivos'] >= (df_consenso['total_votos'] / 2)).astype(int)

# =====================================================================
# 3. FUNDIR COM MAPA E CLASSIFICAR ERROS
# =====================================================================
df_merged = pd.merge(df_consenso, df_index, left_on='id_sample', right_on='name', how='inner')

def classificar_erro(row):
    y = row['y_true']
    p = row['final_threshold_prediction']
    if y == 1 and p == 1: return 'True Positive (TP)'
    if y == 0 and p == 0: return 'True Negative (TN)'
    if y == 0 and p == 1: return 'False Positive (FP)'
    if y == 1 and p == 0: return 'False Negative (FN)'
    return 'Unknown'

df_merged['Classification_Type'] = df_merged.apply(classificar_erro, axis=1)

# =====================================================================
# 4. EXPORTAR PARA GEOPACKAGE (GPKG)
# =====================================================================
print("A converter para formato Geoespacial e a guardar...")
gdf_final = gpd.GeoDataFrame(df_merged, geometry='geometry', crs="EPSG:4326")
gdf_final = gdf_final.drop(columns=['lon', 'lat'])

output_gpkg = os.path.join(OUTPUT_FOLDER, f"consensus_map_year_{ANO_ALVO}.gpkg")

gdf_final.to_file(output_gpkg, driver="GPKG", layer=f"consensus_{ANO_ALVO}")

print(f"\n-> SUCESSO! Mapa definitivo gerado: {output_gpkg}")
print(f"Total de fazendas únicas mapeadas: {len(gdf_final)}")
print("\nResumo do Consenso (Votação Maioritária):")
print(gdf_final['Classification_Type'].value_counts().to_string())