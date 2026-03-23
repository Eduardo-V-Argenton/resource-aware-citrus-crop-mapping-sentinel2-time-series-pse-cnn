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
distancia_uniao_metros = 20

# Tamanho do bloco do Grid (2000 pixels = ~60x60 km no MapBiomas)
TAMANHO_BLOCO = 2000 

classes_interesse = {
    47: 1, # Citrus (Alvo)
    20: 0, # Cana-de-açúcar
    46: 0, # Café
    15: 0, # Pastagem
    9:  0, # Silvicultura
    39: 0, # Soja
    3:  0  # Formação Florestal
}

cidades_alvo = [
    'Aguaí', 'Águas da Prata', 'Casa Branca', 'Espírito Santo do Pinhal', 'Santa Cruz das Palmeiras',
    'Santo Antônio do Jardim', 'São João da Boa Vista', 'Tambaú', 'Vargem Grande do Sul', 
    'Estiva Gerbi', 'Itapira', 'Mogi Guaçu', 'Mogi Mirim'
]

# =====================================================================
# 2. PREPARAÇÃO DO MAPA DA ÁREA DE ESTUDO (POLO LESTE)
# =====================================================================
print("-> Carregando o mapa de municípios do IBGE e isolando")
gdf_mun = gpd.read_file(ficheiro_municipios)
gdf_mun_filtrado = gdf_mun[gdf_mun['NM_MUN'].isin(cidades_alvo)].copy()

# Projetamos para a mesma malha métrica das fazendas para o cruzamento ser instantâneo
gdf_mun_filtrado = gdf_mun_filtrado.to_crs(epsg=6933)

mascara_regiao_alvo = gdf_mun_filtrado.geometry.unary_union

# =====================================================================
# 3. FUNÇÃO DE EXTRAÇÃO 
# =====================================================================
def extrair_e_limpar_poligonos(imagem_bloco, transform_bloco, crs_original, classe_alvo, label_ia):
    # 1. Cria a máscara booleana bruta da classe
    mascara = (imagem_bloco == classe_alvo)
    
    if not mascara.any():
        return None
        
    # Morfologia Matemática na Matriz (Super Rápido)
    estrutura = np.ones((3, 3), dtype=int)
    mascara_limpa = binary_closing(mascara, structure=estrutura)
    
    # 2. Transforma em vetor JÁ LIMPO E UNIDO 
    gerador_poligonos = shapes(mascara_limpa.astype('uint8'), mask=mascara_limpa, transform=transform_bloco)
    
    features = [{'geometry': geom, 'properties': {'mapbiomas_class': classe_alvo, 'label': label_ia}} 
                for geom, valor in gerador_poligonos]
    
    if not features:
        return None
        
    # 3. GeoPandas: Cálculo de área e filtro
    gdf = gpd.GeoDataFrame.from_features(features, crs=crs_original)
    
    # Projeta para metros para calcular a área real
    gdf_metric = gdf.to_crs(epsg=6933)
    gdf_metric['area_ha'] = gdf_metric.geometry.area / 10000.0
    
    # Filtra por tamanho
    gdf_filtrado = gdf_metric[gdf_metric['area_ha'] >= area_minima_ha].copy()
    
    if gdf_filtrado.empty:
        return None
        
    # Simplifica a geometria para deixar o arquivo final leve
    gdf_filtrado['geometry'] = gdf_filtrado.geometry.simplify(tolerance=15, preserve_topology=True)
    
    return gdf_filtrado

# =====================================================================
# 4. LOOP PRINCIPAL COM GRIDDING E FILTRO ESPACIAL
# =====================================================================
files = [f for f in os.listdir(base_dir) if f.endswith('.tif')]
print(files)
for file in files:
    print(f"\n{'='*60}")
    print(f"Processando Ano/Arquivo: {file}")
    
    try:
        with rasterio.open(f"{base_dir}{file}") as src:
            crs_original = src.crs
            largura_total = src.width
            altura_total = src.height
            
            gdfs_do_ano = [] # Vai guardar os resultados da área de estudo
            
            # Navegando pela Malha (Grid)
            for row in range(0, altura_total, TAMANHO_BLOCO):
                for col in range(0, largura_total, TAMANHO_BLOCO):
                    
                    window_width = min(TAMANHO_BLOCO, largura_total - col)
                    window_height = min(TAMANHO_BLOCO, altura_total - row)
                    
                    janela = Window(col, row, window_width, window_height)
                    transform_janela = rasterio.windows.transform(janela, src.transform)
                    
                    imagem_bloco = src.read(1, window=janela)
                    lista_gdfs_bloco = []
                    
                    # Passo A: Extrai todas as classes do Bloco
                    for classe_alvo, label_ia in classes_interesse.items():
                        gdf_classe = extrair_e_limpar_poligonos(imagem_bloco, transform_janela, crs_original, classe_alvo, label_ia)
                        if gdf_classe is not None:
                            lista_gdfs_bloco.append(gdf_classe)
                    
                    if not lista_gdfs_bloco:
                        continue 
                        
                    gdf_mesclado_bloco = gpd.GeoDataFrame(pd.concat(lista_gdfs_bloco, ignore_index=True), crs=6933)
                    
                    # =================================================================
                    # A REGRA DOS 80%: FILTRO INTELIGENTE DE BORDAS
                    # =================================================================
                    # 1. Anota o tamanho original perfeito do polígono
                    gdf_mesclado_bloco['area_original'] = gdf_mesclado_bloco.geometry.area
                    
                    # 2. Calcula matematicamente o "pedaço" do polígono que está dentro da máscara das 17 cidades
                    pedacos_dentro = gdf_mesclado_bloco.geometry.intersection(mascara_regiao_alvo)
                    
                    # 3. Qual foi a porcentagem que ficou lá dentro?
                    porcentagem_dentro = pedacos_dentro.area / gdf_mesclado_bloco['area_original']
                    
                    # 4. Mantém o polígono ORIGINAL (intacto, sem cortes), SÓ SE 80% ou mais dele estiver na nossa região
                    gdf_mesclado_bloco = gdf_mesclado_bloco[porcentagem_dentro >= 0.80].copy()
                    
                    if gdf_mesclado_bloco.empty:
                        # Nenhum polígono atendeu à regra neste bloco. Ignora e pula!
                        continue
                    
                    # Passo B: Balanceamento Ultra-Local Focado
                    positivos = gdf_mesclado_bloco[gdf_mesclado_bloco['label'] == 1]
                    negativos = gdf_mesclado_bloco[gdf_mesclado_bloco['label'] == 0]
                    
                    qtd_positivos = len(positivos)
                    qtd_negativos_total = len(negativos)
                    
                    # Só balanceamos se tiver Laranja DENTRO das 17 cidades neste bloco
                    if qtd_positivos == 0:
                        continue
                    
                    if qtd_negativos_total == 0:
                        gdf_balanceado = positivos
                    else:
                        # A proporção real agora reflete EXATAMENTE a dinâmica da região!
                        proporcoes_reais = negativos['mapbiomas_class'].value_counts(normalize=True)
                        negativos_selecionados = []
                        
                        for classe_mapbiomas, proporcao in proporcoes_reais.items():
                            quantidade_necessaria = int(round(qtd_positivos * proporcao))
                            filtro_classe = negativos[negativos['mapbiomas_class'] == classe_mapbiomas]
                            
                            if quantidade_necessaria > 0:
                                amostra = filtro_classe.sample(n=min(len(filtro_classe), quantidade_necessaria), random_state=42)
                                negativos_selecionados.append(amostra)
                        
                        if negativos_selecionados:
                            negativos_balanceados = pd.concat(negativos_selecionados, ignore_index=True)
                            gdf_balanceado = gpd.GeoDataFrame(
                                pd.concat([positivos, negativos_balanceados], ignore_index=True), 
                                crs=6933
                            )
                        else:
                            gdf_balanceado = positivos
                            
                    gdfs_do_ano.append(gdf_balanceado)
                    print(f"   -> Bloco Polo Leste processado. Polígonos extraídos: {len(gdf_balanceado)}")

            # =================================================================
            # 5. SALVAR O ANO INTEIRO FOCADO NA REGIÃO
            # =================================================================
            if gdfs_do_ano:
                print(f"\n-> Mesclando e salvando dados do Polo Leste para o ano...")
                gdf_estado = gpd.GeoDataFrame(pd.concat(gdfs_do_ano, ignore_index=True), crs=6933)
                
                # Volta para Latitude/Longitude (WGS84) para compatibilidade com AWS
                gdf_final = gdf_estado.to_crs(epsg=4326)
                nome_saida = file.replace('.tif', '_polo_leste.gpkg')
                gdf_final.to_file(f"{output_dir}{nome_saida}", driver="GPKG")
                print(f"-> SUCESSO! Arquivo regional salvo: {nome_saida} (Total: {len(gdf_final)} polígonos)")
            else:
                print("-> Nenhuma cultura de interesse encontrada na Área de Estudo para este ano.")
                
    except Exception as e:
        print(f"Erro ao processar {file}: {e}")