import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
import geopandas as gpd
import pandas as pd
import os
import warnings
import numpy as np
from scipy.ndimage import binary_closing

warnings.filterwarnings("ignore")

# =====================================================================
# 1. CONFIGURAÇÕES E PARÂMETROS
# =====================================================================
base_dir = 'dataset/TIF/'
output_dir = 'dataset/GPKG/'
ficheiro_municipios = 'dataset/SP_MUN_2024/SP_Municipios_2024.shp'

os.makedirs(output_dir, exist_ok=True)

area_minima_ha = 8.0

# Tamanho do bloco do Grid (2000 pixels = ~60x60 km no MapBiomas)
TAMANHO_BLOCO = 2000 

classes_interesse = {
    47: 1, # Citrus (Alvo)
    20: 0, # Cana-de-açúcar
    46: 0, # Café
    15: 0, # Pastagem
    9:  0, # Silvicultura
    39: 0, # Soja
    3:  0,  # Formação Florestal
    11: 0 # Campo Alagado e Área Pantanosa
}

cidades_alvo = [
    'Aguaí', 'Águas da Prata', 'Casa Branca', 'Espírito Santo do Pinhal', 'Santa Cruz das Palmeiras',
    'Santo Antônio do Jardim', 'São João da Boa Vista', 'Tambaú', 'Vargem Grande do Sul', 
    'Estiva Gerbi', 'Itapira', 'Mogi Guaçu', 'Mogi Mirim'
]

# =====================================================================
# 2. PREPARAÇÃO DO MAPA DA ÁREA DE ESTUDO (POLO LESTE)
# =====================================================================
print("-> Carregando o mapa de municípios do IBGE e isolando a região alvo...")
gdf_mun = gpd.read_file(ficheiro_municipios)
gdf_mun_filtrado = gdf_mun[gdf_mun['NM_MUN'].isin(cidades_alvo)].copy()

# Projetamos para a malha métrica (Equal Area) para o cruzamento ser perfeito
gdf_mun_filtrado = gdf_mun_filtrado.to_crs(epsg=6933)
mascara_regiao_alvo = gdf_mun_filtrado.geometry.unary_union

# =====================================================================
# 3. FUNÇÃO DE EXTRAÇÃO (APENAS BRUTA, SEM FILTRO DE ÁREA AQUI)
# =====================================================================
def extrair_poligonos_brutos(imagem_bloco, transform_bloco, crs_original, classe_alvo, label_ia):
    # Cria a máscara booleana da classe
    mascara = (imagem_bloco == classe_alvo)
    
    if not mascara.any():
        return None
        
    # Morfologia Matemática (Remove pequenos buracos/ruídos na imagem)
    estrutura = np.ones((3, 3), dtype=int)
    mascara_limpa = binary_closing(mascara, structure=estrutura)
    
    # Transforma em vetor
    gerador_poligonos = shapes(mascara_limpa.astype('uint8'), mask=mascara_limpa, transform=transform_bloco)
    
    features = [{'geometry': geom, 'properties': {'mapbiomas_class': classe_alvo, 'label': label_ia}} 
                for geom, valor in gerador_poligonos]
    
    if not features:
        return None
        
    # Retorna o GeoDataFrame cru, sem simplificar nem calcular área ainda
    return gpd.GeoDataFrame.from_features(features, crs=crs_original)

# =====================================================================
# 4. LOOP PRINCIPAL COM GRIDDING E COSTURA GEOMÉTRICA
# =====================================================================
files = [f for f in os.listdir(base_dir) if f.endswith('.tif')]
print(f"Arquivos encontrados: {files}")

for file in files:
    print(f"\n{'='*70}")
    print(f"Processando Ano/Arquivo: {file}")
    
    try:
        with rasterio.open(f"{base_dir}{file}") as src:
            crs_original = src.crs
            largura_total = src.width
            altura_total = src.height
            
            lista_todos_poligonos_estado = []
            
            print(" -> Varrendo a malha e extraindo geometria bruta...")
            for row in range(0, altura_total, TAMANHO_BLOCO):
                for col in range(0, largura_total, TAMANHO_BLOCO):
                    
                    window_width = min(TAMANHO_BLOCO, largura_total - col)
                    window_height = min(TAMANHO_BLOCO, altura_total - row)
                    
                    janela = Window(col, row, window_width, window_height)
                    transform_janela = rasterio.windows.transform(janela, src.transform)
                    
                    imagem_bloco = src.read(1, window=janela)
                    
                    for classe_alvo, label_ia in classes_interesse.items():
                        gdf_classe = extrair_poligonos_brutos(imagem_bloco, transform_janela, crs_original, classe_alvo, label_ia)
                        if gdf_classe is not None:
                            lista_todos_poligonos_estado.append(gdf_classe)
            
            if not lista_todos_poligonos_estado:
                print(" -> Nenhuma geometria encontrada no estado inteiro. Pulando...")
                continue 
                
            # Junta todos os pedaços extraídos do estado de SP
            gdf_estado_bruto = gpd.GeoDataFrame(pd.concat(lista_todos_poligonos_estado, ignore_index=True), crs=crs_original)
            
            # Projeta para metros para iniciar a matemática espacial
            gdf_estado_bruto = gdf_estado_bruto.to_crs(epsg=6933)
            
            # =================================================================
            # PENEIRA 1: SJOIN (FILTRO RÁPIDO PARA DESCARTAR O RESTO DO ESTADO)
            # =================================================================
            print(" -> Filtrando apenas fazendas do Polo Leste (Spatial Join Rápido)...")
            gdf_polo = gpd.sjoin(gdf_estado_bruto, gdf_mun_filtrado[['NM_MUN', 'geometry']], how="inner", predicate="intersects")
            
            if gdf_polo.empty:
                print(" -> Nenhum polígono tocou as cidades alvo. Pulando...")
                continue

            # =================================================================
            # 3. COSTURAR AS FAZENDAS (RESOLVENDO A GUILHOTINA)
            # =================================================================
            print(" -> Costurando polígonos que foram cortados pela malha do Grid...")
            gdf_polo['geometry'] = gdf_polo.geometry.buffer(0.1) 
            
            # Dissolve apenas pelas classes. A coluna NM_MUN será removida aqui para evitar conflitos
            gdf_costurado = gdf_polo.dissolve(by=['mapbiomas_class', 'label']).explode(index_parts=False).reset_index()
            
            gdf_costurado['geometry'] = gdf_costurado.geometry.buffer(-0.1) 
            
            # =================================================================
            # 4. FILTRO DE ÁREA E REGRA DOS 80%
            # =================================================================
            print(" -> Aplicando filtro de área mínima (8.0 ha)...")
            gdf_costurado['area_ha'] = gdf_costurado.geometry.area / 10000.0
            gdf_costurado = gdf_costurado[gdf_costurado['area_ha'] >= area_minima_ha].copy()
            
            if gdf_costurado.empty:
                print(" -> Nenhum polígono sobreviveu ao filtro de área.")
                continue
                
            print(" -> Calculando a Regra dos 80% de Intersecção...")
            gdf_costurado['area_original'] = gdf_costurado.geometry.area
            pedacos_dentro = gdf_costurado.geometry.intersection(mascara_regiao_alvo)
            porcentagem_dentro = pedacos_dentro.area / gdf_costurado['area_original']
            
            gdf_valido = gdf_costurado[porcentagem_dentro >= 0.80].copy()
            
            # =================================================================
            # 4.5. UNDERSAMPLING ESTRATIFICADO (O "HACK" DO DISCO RÍGIDO)
            # =================================================================
            if not gdf_valido.empty:
                print(" -> Aplicando Amostragem Estratificada (Proporção 1:4)...")
                
                # Separa a minoria (Laranja) da maioria (Salada de Culturas)
                gdf_citrus = gdf_valido[gdf_valido['label'] == 1].copy()
                gdf_outros = gdf_valido[gdf_valido['label'] == 0].copy()
                
                qtd_citrus = len(gdf_citrus)
                
                if qtd_citrus > 0:
                    # Limite de 4 Outros para cada 1 Citrus
                    qtd_outros_desejada = qtd_citrus * 4 
                    
                    if len(gdf_outros) > qtd_outros_desejada:
                        # Corta a classe 0 de forma proporcional para manter a diversidade do fundo
                        fracao = qtd_outros_desejada / len(gdf_outros)
                        gdf_outros_reduzido = gdf_outros.groupby(['mapbiomas_class', 'NM_MUN'], group_keys=False).apply(
                            lambda x: x.sample(frac=fracao, random_state=42) if len(x) > 0 else x
                        )
                        # Une tudo novamente
                        gdf_valido = gpd.GeoDataFrame(pd.concat([gdf_citrus, gdf_outros_reduzido], ignore_index=True), crs=gdf_costurado.crs)
                        print(f"    [!] Reduzido de {len(gdf_outros)} para {len(gdf_outros_reduzido)} polígonos de fundo.")
                else:
                    # Se num ano/bloco não houver Laranja, retemos apenas um limite de fundo para não perder a referência do ano
                    if len(gdf_outros) > 500:
                        gdf_valido = gdf_outros.sample(n=500, random_state=42)
                        print(f"    [!] Sem Citrus. Fundo limitado a 500 polígonos.")

            # =================================================================
            # 5. SALVAR O GPKG (AGORA LEVE E BALANCEADO EM 1:4)
            # =================================================================
            if not gdf_valido.empty:
                print(f" -> Recuperando nomes das cidades e salvando (Total: {len(gdf_valido)} polígonos)...")
                
                colunas_preservar = ['mapbiomas_class', 'label', 'area_ha', 'geometry']
                gdf_limpo = gdf_valido[colunas_preservar].copy()
                
                # 2. SJOIN FINAL: Para carimbar a cidade na fazenda costurada
                gdf_final_com_cidade = gpd.sjoin(
                    gdf_limpo, 
                    gdf_mun_filtrado[['NM_MUN', 'geometry']], 
                    how="left", 
                    predicate="intersects"
                )
                
                # 3. TRATAMENTO DE DUPLICATAS: Se uma fazenda toca duas cidades, pegamos a primeira
                gdf_final_com_cidade = gdf_final_com_cidade.drop_duplicates(subset='geometry')
                
                # 4. CONVERSÃO E SIMPLIFICAÇÃO
                gdf_final = gdf_final_com_cidade.to_crs(epsg=4326)
                
                # Simplificação leve para o arquivo não ficar gigante (usando graus decimais agora)
                gdf_final['geometry'] = gdf_final.geometry.simplify(tolerance=0.0001, preserve_topology=True)
                
                # 5. FILTRO FINAL DE COLUNAS (Com verificação de segurança)
                colunas_finais = ['mapbiomas_class', 'label', 'area_ha', 'NM_MUN', 'geometry']
                
                gdf_final = gdf_final[colunas_finais]
                
                nome_saida = file.replace('.tif', '.gpkg')
                gdf_final.to_file(f"{output_dir}{nome_saida}", driver="GPKG")
                print(f" -> SUCESSO! Arquivo salvo: {nome_saida}")
            else:
                print(" -> Nenhum polígono sobreviveu à Regra dos 80%.")
                
    except Exception as e:
        print(f"Erro ao processar {file}: {e}")