
import os
import geopandas as gpd

# =====================================================================
# 1. CONFIGURAÇÕES INICIAIS
# =====================================================================
base_dir = 'dataset/GPKG/' # Apontando para os vetores agora

cidades_alvo = [
    'Aguaí', 'Águas da Prata', 'Casa Branca', 'Espírito Santo do Pinhal', 'Santa Cruz das Palmeiras',
    'Santo Antônio do Jardim', 'São João da Boa Vista', 'Tambaú', 'Vargem Grande do Sul', 
    'Estiva Gerbi', 'Itapira', 'Mogi Guaçu', 'Mogi Mirim'
]

files = [f for f in os.listdir(base_dir) if f.endswith('.gpkg')]

for file in files:
    print(f"\n{'='*60}")
    print(f"Analisando a realidade do solo no arquivo: {file}")
    
    caminho_gpkg = os.path.join(base_dir, file)
    
    try:
        # 1. Carrega o GeoPackage
        gdf = gpd.read_file(caminho_gpkg)
        
        # 2. Filtro de segurança: garante que só analisamos as cidades alvo
        # (Caso o GPKG tenha pego alguma cidade de borda a mais)
        if 'NM_MUN' in gdf.columns:
            gdf = gdf[gdf['NM_MUN'].isin(cidades_alvo)].copy()
            
        if gdf.empty:
            print(" -> Nenhum polígono encontrado para as cidades alvo neste arquivo.")
            continue

        # 3. Matemática da Área (Projetamos para a malha métrica EPSG:6933)
        # Isso garante precisão absoluta da área (1 hectare = 10.000 m²)
        gdf_projetado = gdf.to_crs(epsg=6933)
        gdf['area_calc_ha'] = gdf_projetado.geometry.area / 10000.0
        
        # 4. Agrupa pelas classes e soma as áreas
        df_agrupado = gdf.groupby('mapbiomas_class')['area_calc_ha'].sum().reset_index()
        df_agrupado.rename(columns={'mapbiomas_class': 'Código', 'area_calc_ha': 'Área (ha)'}, inplace=True)
        
        # 5. Calcula as porcentagens
        area_total_regiao = df_agrupado['Área (ha)'].sum()
        df_agrupado['Porcentagem (%)'] = (df_agrupado['Área (ha)'] / area_total_regiao) * 100
        
        # 6. Ordena e formata
        df_resultados = df_agrupado.sort_values(by='Área (ha)', ascending=False).reset_index(drop=True)
        
        df_resultados['Área (ha)'] = df_resultados['Área (ha)'].map('{:,.2f}'.format)
        df_resultados['Porcentagem (%)'] = df_resultados['Porcentagem (%)'].map('{:.2f}%'.format)
        
        print(f"\nRanking de Distribuição Espacial ({file}):")
        print(df_resultados.to_string(index=False))

    except Exception as e:
        print(f"Erro ao processar {file}: {e}")
