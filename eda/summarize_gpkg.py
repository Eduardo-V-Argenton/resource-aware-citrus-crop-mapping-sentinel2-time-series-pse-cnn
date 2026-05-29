import os
import pandas as pd
import geopandas as gpd

# =====================================================================
# CONFIGs
# =====================================================================
gpkg_dir = 'dataset/GPKG/'
CLASSE_CITRUS = 47  

resultados_gerais = []
areas_por_classe_list = []

# =====================================================================
# PROCESSAMENTO
# =====================================================================
print("-> Lendo arquivos GPKG e calculando áreas...\n")

arquivos_gpkg = [f for f in os.listdir(gpkg_dir) if f.endswith('.gpkg')]

for file in arquivos_gpkg:
    filepath = os.path.join(gpkg_dir, file)
    
    # Lê o vetor
    gdf = gpd.read_file(filepath)
    
    if gdf.empty:
        print(f"[{file}] Arquivo vazio, pulando...")
        continue

    # Verifica se a coluna de região existe no arquivo
    if 'NM_RGI' not in gdf.columns:
        print(f"[{file}] ATENÇÃO: Coluna 'NM_RGI' não encontrada. Pulando...")
        continue

    # Agrupa por REGIÃO e por CLASSE simultaneamente
    # Assim garantimos que as cidades não se misturem
    df_agrupado = gdf.groupby(['NM_RGI', 'mapbiomas_class'])['area_ha'].sum().reset_index()
    
    # Pega a lista de regiões únicas que existem dentro deste arquivo específico
    regioes_no_arquivo = df_agrupado['NM_RGI'].unique()
    
    for regiao in regioes_no_arquivo:
        # Isola os dados apenas desta região
        df_regiao = df_agrupado[df_agrupado['NM_RGI'] == regiao]
        
        # 1. Área Total da Região
        area_total = df_regiao['area_ha'].sum()
        
        # 2. Área de Citrus na Região
        df_citrus = df_regiao[df_regiao['mapbiomas_class'] == CLASSE_CITRUS]
        area_citrus = df_citrus['area_ha'].sum() if not df_citrus.empty else 0.0
        
        # 3. Proporções
        area_outros = area_total - area_citrus
        razao_citrus_outros = (area_citrus / area_outros) if area_outros > 0 else 0.0
        porcentagem_citrus = (area_citrus / area_total) * 100 if area_total > 0 else 0.0
        
        resultados_gerais.append({
            'Arquivo': file,
            'Regiao': regiao,
            'Area_Total_ha': round(area_total, 2),
            'Area_Citrus_ha': round(area_citrus, 2),
            'Area_Outros_ha': round(area_outros, 2),
            'Razao_Citrus_Outros': round(razao_citrus_outros, 4),
            'Porcentagem_Citrus_%': round(porcentagem_citrus, 2)
        })
        
        # Guarda o detalhamento por classe para a tabela secundária
        for _, row in df_regiao.iterrows():
            areas_por_classe_list.append({
                'Arquivo': file,
                'Regiao': regiao,
                'mapbiomas_class': row['mapbiomas_class'],
                'area_ha': row['area_ha']
            })

# =====================================================================
# RESULTADOS
# =====================================================================
df_resumo = pd.DataFrame(resultados_gerais)
df_classes = pd.DataFrame(areas_por_classe_list)

print("="*85)
print("RESUMO POR REGIÃO/ARQUIVO (TOTAL, CITRUS E PROPORÇÕES)")
print("="*85)
# Ordena por Região e Arquivo para ficar fácil de ler a série temporal de cada local
df_resumo = df_resumo.sort_values(by=['Regiao', 'Arquivo']).reset_index(drop=True)
print(df_resumo.to_string())

print("\n" + "="*85)
print("ÁREA DETALHADA POR CLASSE MAPBIOMAS (Amostra Citrus)")
print("="*85)
df_citrus_only = df_classes[df_classes['mapbiomas_class'] == CLASSE_CITRUS].sort_values(by=['Regiao', 'Arquivo'])
print(df_citrus_only.to_string() if not df_citrus_only.empty else "Nenhuma área de Citrus encontrada.")