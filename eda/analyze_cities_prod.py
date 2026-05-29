#https://sidra.ibge.gov.br/Tabela/5457
# Área plantada ou destinada à colheita (Hectares [1988 a 2024])

# https://www.ibge.gov.br/geociencias/organizacao-do-territorio/divisao-regional/15778-divisoes-regionais-do-brasil.html?=&t=acesso-ao-produto
# 
import pandas as pd
import geopandas as gpd

df_raw = pd.read_csv('eda/area_plantada.csv', skiprows=4, header=None)
city_list = df_raw.iloc[1:, 0].reset_index(drop=True)
planted_area = df_raw.iloc[1:, 1:].reset_index(drop=True)
planted_area.replace(['...', '-'], [None, 0], inplace=True)
planted_area = planted_area.apply(pd.to_numeric, errors='coerce').fillna(0)

total_citrus_por_cidade = planted_area.iloc[:, :24].sum(axis=1)

df_total_cidade = pd.DataFrame({
    "city": city_list,
    "producao_total_6_anos": total_citrus_por_cidade
}).dropna(subset=['city'])

df_total_cidade['city_clean'] = df_total_cidade['city'].str.replace(r'\s*\([A-Z]{2}\)', '', regex=True).str.strip()

df_ibge = pd.read_excel('eda/regioes_geograficas_composicao_por_municipios_2017_20180911.ods', engine='odf')

df_merged = pd.merge(
    df_total_cidade, 
    df_ibge, 
    left_on='city_clean', 
    right_on='nome_mun', 
    how='inner'
)

producao_total_regiao = df_merged.groupby('nome_rgi')['producao_total_6_anos'].sum().reset_index()
producao_total_regiao['media_anual_regiao'] = producao_total_regiao['producao_total_6_anos'] / 6
gdf_rgi = gpd.read_file("eda/RG2017_rgi_20180911/RG2017_rgi.shp")
gdf_rgi = gdf_rgi.to_crs("EPSG:5880")
gdf_rgi["area_m2"] = gdf_rgi.geometry.area
gdf_rgi["area_ha"] = gdf_rgi["area_m2"] / 10000
gdf_clean = gdf_rgi[["nome_rgi","area_ha"]]
producao_merged = pd.merge(
    producao_total_regiao,
    gdf_clean,
    left_on='nome_rgi',
    right_on='nome_rgi',
    how="inner"
)
producao_merged['proporcao_citrus'] = producao_merged['media_anual_regiao'] / producao_merged["area_ha"]
ranking_regioes = producao_merged.sort_values(by='proporcao_citrus', ascending=False).reset_index(drop=True)

ranking_regioes = ranking_regioes[['nome_rgi', 'media_anual_regiao', 'proporcao_citrus', 'area_ha']]
ranking_regioes.columns = ['Região Imediata', 'Média Anual', 'Proporcao citrus', 'Area']
print(ranking_regioes.head(10))