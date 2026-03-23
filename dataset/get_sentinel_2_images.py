import geopandas as gpd
import pandas as pd
import numpy as np
import os
import time
import warnings
import requests
import random
from rasterio.io import MemoryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import box

import ee

warnings.filterwarnings("ignore")

# =====================================================================
# 1. INICIALIZAÇÃO DO GOOGLE EARTH ENGINE (GEE)
# =====================================================================
# Lembre-se de colocar o seu ID do projeto aqui!
MEU_PROJETO_GEE = 'clean-bindery-462116-u8' 

print("-> Conectando ao Google Earth Engine...")
try:
    ee.Initialize(project=MEU_PROJETO_GEE, opt_url='https://earthengine-highvolume.googleapis.com')
except Exception:
    print("-> Autenticação necessária. A abrir o navegador...")
    ee.Authenticate()
    ee.Initialize(project=MEU_PROJETO_GEE, opt_url='https://earthengine-highvolume.googleapis.com')

ficheiro_municipios = 'dataset/SP_MUN_2024/SP_Municipios_2024.shp'
pasta_tensores = 'dataset/Tensores_Treino/'
ficheiro_csv = 'dataset/dataset_index.csv'

os.makedirs(pasta_tensores, exist_ok=True)

nomes_culturas = {
    47: 'Citrus', 20: 'Cana-de-açúcar', 46: 'Café',
    15: 'Pastagem', 9:  'Silvicultura', 39: 'Soja', 3:  'Formação Florestal'
}

cidades_alvo = [
    'Aguaí', 'Águas da Prata', 'Casa Branca', 'Espírito Santo do Pinhal', 'Santa Cruz das Palmeiras',
    'Santo Antônio do Jardim', 'São João da Boa Vista', 'Tambaú', 'Vargem Grande do Sul', 
    'Estiva Gerbi', 'Itapira', 'Mogi Guaçu', 'Mogi Mirim'
]

# =====================================================================
# 2. MOTOR DE EXTRAÇÃO NO GEE
# =====================================================================
def baixar_serie_temporal_gee(bbox_fazenda, ano_imagem, index_fazenda, parte_nome, max_tentativas=3):
    geom = ee.Geometry.Rectangle(bbox_fazenda)
    
    datas = pd.date_range(start=f'{ano_imagem}-01-01', end=f'{ano_imagem}-12-31', freq='15D')
    periodos = [(datas[i].strftime('%Y-%m-%d'), datas[i+1].strftime('%Y-%m-%d')) for i in range(len(datas)-1)]
    periodos.append((datas[-1].strftime('%Y-%m-%d'), f'{ano_imagem}-12-31'))
    num_periodos = len(periodos)

    base_10m = ee.Image.constant([0, 0, 0, 0]).rename(['B2', 'B3', 'B4', 'B8']).toFloat()
    base_20m = ee.Image.constant([0, 0, 0, 0, 0, 0]).rename(['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']).toFloat()

    lista_10m = []
    lista_20m = []

    for inicio, fim in periodos:
        col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(geom) \
                .filterDate(inicio, fim)
        
        def mascara_nuvens_scl(img):
            scl = img.select('SCL')
            mascara_limpa = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
            return img.updateMask(mascara_limpa)
        
        def processar_colecao(colecao):
            mediana = colecao.map(mascara_nuvens_scl).median()
            img_10 = mediana.select(['B2', 'B3', 'B4', 'B8']).unmask(0).toFloat()
            img_20 = mediana.select(['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']).unmask(0).toFloat()
            return ee.Dictionary({'10m': img_10, '20m': img_20})
            
        resultado = ee.Algorithms.If(
            col.size().gt(0),
            processar_colecao(col),
            ee.Dictionary({'10m': base_10m, '20m': base_20m})
        )
        
        dict_res = ee.Dictionary(resultado)
        lista_10m.append(ee.Image(dict_res.get('10m')))
        lista_20m.append(ee.Image(dict_res.get('20m')))

    stack_10m = ee.ImageCollection(lista_10m).toBands()
    stack_20m = ee.ImageCollection(lista_20m).toBands()

    for tentativa in range(max_tentativas):
        try:
            url_10m = stack_10m.getDownloadURL({'region': geom, 'scale': 10, 'format': 'GEO_TIFF'})
            resp_10m = requests.get(url_10m, timeout=60)
            with MemoryFile(resp_10m.content) as memfile:
                with memfile.open() as src:
                    arr_10m_flat = src.read() 
            
            url_20m = stack_20m.getDownloadURL({'region': geom, 'scale': 20, 'format': 'GEO_TIFF'})
            resp_20m = requests.get(url_20m, timeout=60)
            with MemoryFile(resp_20m.content) as memfile:
                with memfile.open() as src:
                    arr_20m_flat = src.read() 
            
            _, H10, W10 = arr_10m_flat.shape
            tensor_10m = arr_10m_flat.reshape((num_periodos, 4, H10, W10))
            
            _, H20, W20 = arr_20m_flat.shape
            tensor_20m = arr_20m_flat.reshape((num_periodos, 6, H20, W20))
            
            return tensor_10m, tensor_20m
            
        except Exception as e:
            msg = str(e)[:60].replace('\n', ' ')
            print(f"      [AVISO GEE] Sobrecarga em {parte_nome} (Tentar {tentativa+1}/3): {msg}...")
            time.sleep((2 ** tentativa) + random.uniform(0, 1))

    return None, None

# =====================================================================
# 3. FUNÇÃO FATIADORA DE FAZENDAS GIGANTES
# =====================================================================
def processar_fazenda(index, linha, ano_imagem):
    classe_ia = linha['label']
    classe_mapbiomas = linha['mapbiomas_class']
    cidade = str(linha['NM_MUN']) 
    area_total = linha['area_ha']
    nome_cultura = nomes_culturas.get(classe_mapbiomas, 'Outra Cultura')
    geometria_real = linha.geometry
    
    # O limite seguro que a Google aceita numa única tacada (~1.5 km x 1.5 km)
    MAX_GRAUS = 0.015 
    
    minx, miny, maxx, maxy = geometria_real.bounds
    largura = maxx - minx
    altura = maxy - miny
    
    resultados = []
    
    # Descobre quantas fatias teremos que fazer para cobrir a fazenda toda
    passos_x = int(np.ceil(largura / MAX_GRAUS))
    passos_y = int(np.ceil(altura / MAX_GRAUS))
    
    contador_parte = 0
    
    for i in range(passos_x):
        for j in range(passos_y):
            sub_minx = minx + i * MAX_GRAUS
            sub_maxx = min(minx + (i + 1) * MAX_GRAUS, maxx)
            sub_miny = miny + j * MAX_GRAUS
            sub_maxy = min(miny + (j + 1) * MAX_GRAUS, maxy)
            
            caixa_corte = box(sub_minx, sub_miny, sub_maxx, sub_maxy)
            
            # Só extraímos o pedaço se ele realmente cruzar com o desenho da fazenda original
            if geometria_real.intersects(caixa_corte):
                contador_parte += 1
                
                # Se for uma fazenda pequena que não precisou de cortes, não coloca o "p01" no nome
                if passos_x == 1 and passos_y == 1:
                    nome_base = f"img_{index:05d}_{ano_imagem}"
                else:
                    nome_base = f"img_{index:05d}_p{contador_parte:02d}_{ano_imagem}"
                    
                caminho_10m = os.path.join(pasta_tensores, f"{nome_base}_10m.npy")
                caminho_20m = os.path.join(pasta_tensores, f"{nome_base}_20m.npy")
                
                # Sistema Anti-Quebra
                if os.path.exists(caminho_10m) and os.path.exists(caminho_20m):
                    resultados.append({'status': 'pulado', 'id': index, 'nome': nome_base})
                    continue

                bbox_sub = [sub_minx, sub_miny, sub_maxx, sub_maxy]
                
                tensor_10m, tensor_20m = baixar_serie_temporal_gee(bbox_sub, ano_imagem, index, nome_base)
                
                if tensor_10m is not None and tensor_20m is not None:
                    np.save(caminho_10m, tensor_10m)
                    np.save(caminho_20m, tensor_20m)
                    
                    dado_csv = {
                        'nome_base': nome_base, 
                        'label_ia': classe_ia,             
                        'classe_mapbiomas': classe_mapbiomas, 
                        'cultura_real': nome_cultura,      
                        'cidade': cidade,
                        'ano': ano_imagem,
                        'area_ha': round(area_total, 2),
                        'e_pedaco': 1 if (passos_x > 1 or passos_y > 1) else 0
                    }
                    resultados.append({'status': 'sucesso', 'id': index, 'dados': dado_csv, 'nome': nome_base})
                else:
                    resultados.append({'status': 'erro', 'id': index, 'nome': nome_base})

    return resultados

def adicionar_ao_csv_seguro(registro, caminho_csv):
    df = pd.DataFrame([registro])
    if not os.path.exists(caminho_csv):
        df.to_csv(caminho_csv, index=False, encoding='utf-8')
    else:
        df.to_csv(caminho_csv, mode='a', header=False, index=False, encoding='utf-8')

# =====================================================================
# 4. O GRANDE LOOP DE PROCESSAMENTO MULTITHREAD
# =====================================================================
for ANO_IMAGEM in range(2017, 2025):
    print(f"\n{'='*50}\nA iniciar processamento GEE para o Ano: {ANO_IMAGEM}")
    
    ficheiro_dataset = f'dataset/GPKG/mapbiomas-brazil-collection-101-saopaulosp-{ANO_IMAGEM}.gpkg' 
    if not os.path.exists(ficheiro_dataset):
        print(f"Ficheiro não encontrado. A saltar ano...")
        continue
    
    gdf_mestre = gpd.read_file(ficheiro_dataset).to_crs(epsg=4326)
    gdf_mun = gpd.read_file(ficheiro_municipios).to_crs(epsg=4326)
    
    print(" -> A aplicar filtro rigoroso das Regiões Imediatas...")
    gdf_mun_filtrado = gdf_mun[gdf_mun['NM_MUN'].isin(cidades_alvo)]
    gdf_final = gpd.sjoin(gdf_mestre, gdf_mun_filtrado[['NM_MUN', 'geometry']], how="inner", predicate="intersects")
    
    total_fazendas = len(gdf_final)
    print(f" -> A iniciar Motor Multithread (10 Ligações GEE) para {total_fazendas} matrizes fundiárias...")
    
    sucessos = 0
    pulados = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        tarefas = {executor.submit(processar_fazenda, index, linha, ANO_IMAGEM): index 
                   for index, linha in gdf_final.iterrows()}
        
        for futuro in as_completed(tarefas):
            resultados_fazenda = futuro.result()
            
            # Como uma fazenda pode ter virado vários pedaços, iteramos por eles
            for res in resultados_fazenda:
                if res['status'] == 'pulado':
                    pulados += 1
                elif res['status'] == 'sucesso':
                    sucessos += 1
                    adicionar_ao_csv_seguro(res['dados'], ficheiro_csv)
                    print(f"    [OK] {res['nome']} guardado! ({res['dados']['cultura_real']})")
                elif res['status'] == 'erro':
                    print(f"    [ERRO] Falha crítica no recorte {res['nome']}.")

    print(f"\nResumo GEE do Ano {ANO_IMAGEM}: {sucessos} recortes descarregados, {pulados} já existiam.")