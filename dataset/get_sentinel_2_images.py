import geopandas as gpd
import pandas as pd
import numpy as np
import os
import time
import warnings
import requests
import random
import threading
from rasterio.io import MemoryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import box
import ee

warnings.filterwarnings("ignore")

# =====================================================================
# 1. INICIALIZAÇÃO E CONFIGURAÇÕES GLOBAIS
# =====================================================================
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
csv_lock = threading.Lock()

cidades_alvo = [
    'Aguaí', 'Águas da Prata', 'Casa Branca', 'Espírito Santo do Pinhal', 'Santa Cruz das Palmeiras',
    'Santo Antônio do Jardim', 'São João da Boa Vista', 'Tambaú', 'Vargem Grande do Sul', 
    'Estiva Gerbi', 'Itapira', 'Mogi Guaçu', 'Mogi Mirim'
]

# =====================================================================
# 2. PROTOCOLO DE AUTOCURA E RECONCILIAÇÃO (SELF-HEALING)
# =====================================================================
print("\n" + "="*60)
print(" INICIANDO AUDITORIA E CURA DO DATASET EXISTENTE")
print("="*60)

fazendas_processadas_csv = set()

if os.path.exists(ficheiro_csv):
    try:
        df_audit = pd.read_csv(ficheiro_csv)
        linhas_validas = []
        
        print(" -> Verificando integridade física dos tensores mapeados no CSV...")
        for index, row in df_audit.iterrows():
            nome = row['nome_base']
            c10 = os.path.join(pasta_tensores, f"{nome}_10m.npy")
            c20 = os.path.join(pasta_tensores, f"{nome}_20m.npy")
            
            # Regra de Ouro: Arquivo tem que existir e ser maior que 1 KB (1000 bytes)
            if os.path.exists(c10) and os.path.getsize(c10) > 1000 and \
               os.path.exists(c20) and os.path.getsize(c20) > 1000:
                linhas_validas.append(row)
                fazendas_processadas_csv.add(nome)
        
        df_limpo = pd.DataFrame(linhas_validas)
        
        # Reescreve o CSV apenas com dados 100% perfeitos
        df_limpo.to_csv(ficheiro_csv, index=False)
        print(f" -> Auditoria CSV Concluída: {len(df_audit)} registros originais -> {len(df_limpo)} registros saudáveis.")
        
        print(" -> Caçando e exterminando arquivos órfãos ou corrompidos no HD...")
        apagados = 0
        for f in os.listdir(pasta_tensores):
            if f.endswith('.npy'):
                # Descobre quem é o "dono" do arquivo
                nome_base_arquivo = f.replace('_10m.npy', '').replace('_20m.npy', '')
                
                # Se o dono não está no CSV validado, é lixo. Deleta.
                if nome_base_arquivo not in fazendas_processadas_csv:
                    os.remove(os.path.join(pasta_tensores, f))
                    apagados += 1
        
        print(f" -> Limpeza de HD Concluída: {apagados} arquivos defeituosos foram apagados.")
        
    except Exception as e:
        print(f" -> Erro na auditoria: {e}. O processo continuará com cautela.")
else:
    print(" -> CSV não encontrado. A iniciar um dataset completamente novo.")

# =====================================================================
# 3. MOTOR DE EXTRAÇÃO NO GEE
# =====================================================================
def baixar_serie_temporal_gee(bbox_fazenda, ano_imagem, index_fazenda, parte_nome, max_tentativas=3):
    geom = ee.Geometry.Rectangle(bbox_fazenda)
    datas = pd.date_range(start=f'{ano_imagem}-01-01', end=f'{ano_imagem}-12-31', freq='15D')
    periodos = [(datas[i].strftime('%Y-%m-%d'), datas[i+1].strftime('%Y-%m-%d')) for i in range(len(datas)-1)]
    periodos.append((datas[-1].strftime('%Y-%m-%d'), f'{ano_imagem}-12-31'))
    num_periodos = len(periodos)

    base_10m = ee.Image.constant([0, 0, 0, 0]).rename(['B2', 'B3', 'B4', 'B8']).toFloat()
    base_20m = ee.Image.constant([0, 0, 0, 0, 0, 0]).rename(['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']).toFloat()

    lista_10m, lista_20m = [], []

    for inicio, fim in periodos:
        col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(geom).filterDate(inicio, fim)
        
        def mascara_nuvens_scl(img):
            scl = img.select('SCL')
            mascara_limpa = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
            return img.updateMask(mascara_limpa)
        
        def processar_colecao(colecao):
            mediana = colecao.map(mascara_nuvens_scl).median()
            img_10 = mediana.select(['B2', 'B3', 'B4', 'B8']).unmask(0).toFloat()
            img_20 = mediana.select(['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']).unmask(0).toFloat()
            return ee.Dictionary({'10m': img_10, '20m': img_20})
            
        resultado = ee.Algorithms.If(col.size().gt(0), processar_colecao(col), ee.Dictionary({'10m': base_10m, '20m': base_20m}))
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
            print(f"      [AVISO GEE] Sobrecarga em {parte_nome} (Tentativa {tentativa+1}/3): {msg}...")
            time.sleep((2 ** tentativa) + random.uniform(0, 1))

    return None, None

# =====================================================================
# 4. FUNÇÃO FATIADORA COM ID ESPACIAL E PROTEÇÃO ANTI-QUEBRA
# =====================================================================
def processar_fazenda(index, linha, ano_imagem):
    classe_ia = linha['label']
    classe_mapbiomas = linha['mapbiomas_class']
    cidade = str(linha['NM_MUN']) 
    area_total = linha['area_ha']
    geometria_real = linha.geometry
    
    # ID Espacial persistente
    centro = geometria_real.centroid
    id_espacial = f"loc_{round(centro.x, 4)}_{round(centro.y, 4)}"
    
    # Dicionário dinâmico de culturas
    nomes_culturas = {47: 'Citrus', 20: 'Cana-de-açúcar', 46: 'Café', 15: 'Pastagem', 9: 'Silvicultura', 39: 'Soja', 3: 'Formação Florestal'}
    nome_cultura = nomes_culturas.get(classe_mapbiomas, 'Outra Cultura')
    
    MAX_GRAUS = 0.015 
    minx, miny, maxx, maxy = geometria_real.bounds
    passos_x = int(np.ceil((maxx - minx) / MAX_GRAUS))
    passos_y = int(np.ceil((maxy - miny) / MAX_GRAUS))
    
    resultados = []
    contador_parte = 0
    
    for i in range(passos_x):
        for j in range(passos_y):
            sub_minx = minx + i * MAX_GRAUS
            sub_maxx = min(minx + (i + 1) * MAX_GRAUS, maxx)
            sub_miny = miny + j * MAX_GRAUS
            sub_maxy = min(miny + (j + 1) * MAX_GRAUS, maxy)
            
            caixa_corte = box(sub_minx, sub_miny, sub_maxx, sub_maxy)
            
            if geometria_real.intersects(caixa_corte):
                contador_parte += 1
                nome_base = f"img_{index:05d}_{ano_imagem}" if (passos_x == 1 and passos_y == 1) else f"img_{index:05d}_p{contador_parte:02d}_{ano_imagem}"
                
                # A Barreira Anti-Quebra
                if nome_base in fazendas_processadas_csv:
                    resultados.append({'status': 'pulado', 'id': index, 'nome': nome_base})
                    continue

                caminho_10m = os.path.join(pasta_tensores, f"{nome_base}_10m.npy")
                caminho_20m = os.path.join(pasta_tensores, f"{nome_base}_20m.npy")
                bbox_sub = [sub_minx, sub_miny, sub_maxx, sub_maxy]
                
                tensor_10m, tensor_20m = baixar_serie_temporal_gee(bbox_sub, ano_imagem, index, nome_base)
                
                if tensor_10m is not None and tensor_20m is not None:
                    # Salva os arquivos (Se for corrompido, a auditoria pega na próxima execução)
                    np.save(caminho_10m, tensor_10m)
                    np.save(caminho_20m, tensor_20m)
                    
                    dado_csv = {
                        'id_poligono': id_espacial, 
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

# =====================================================================
# 5. ESCRITA SEGURA NO CSV (COM THREAD LOCK)
# =====================================================================
def adicionar_ao_csv_seguro(registro, caminho_csv):
    df = pd.DataFrame([registro])
    with csv_lock:
        if not os.path.exists(caminho_csv):
            df.to_csv(caminho_csv, index=False, encoding='utf-8')
        else:
            df.to_csv(caminho_csv, mode='a', header=False, index=False, encoding='utf-8')
        # Adiciona ao set global para o caso de a mesma fazenda ser chamada de novo na mesma execução
        fazendas_processadas_csv.add(registro['nome_base'])

# =====================================================================
# 6. O GRANDE LOOP DE PROCESSAMENTO MULTITHREAD
# =====================================================================
for ANO_IMAGEM in range(2019, 2025):
    print(f"\n{'='*50}\nA iniciar processamento GEE para o Ano: {ANO_IMAGEM}")
    
    ficheiro_dataset = f'dataset/GPKG/mapbiomas-brazil-collection-101-saopaulosp-{ANO_IMAGEM}.gpkg' 
    if not os.path.exists(ficheiro_dataset):
        print(f"Ficheiro {ficheiro_dataset} não encontrado. A saltar ano...")
        continue
    
    gdf_mestre = gpd.read_file(ficheiro_dataset).to_crs(epsg=4326)
    
    if 'NM_MUN' not in gdf_mestre.columns:
        print(" -> Coluna NM_MUN não encontrada no GPKG. Aplicando Join espacial de segurança...")
        gdf_mun = gpd.read_file(ficheiro_municipios).to_crs(epsg=4326)
        gdf_mun_filtrado = gdf_mun[gdf_mun['NM_MUN'].isin(cidades_alvo)]
        gdf_final = gpd.sjoin(gdf_mestre, gdf_mun_filtrado[['NM_MUN', 'geometry']], how="left", predicate="intersects")
    else:
        gdf_final = gdf_mestre

    gdf_final = gdf_final.drop_duplicates(subset='geometry').reset_index(drop=True)
    
    total_fazendas = len(gdf_final)
    print(f" -> A iniciar Motor Multithread (20 Ligações GEE) para {total_fazendas} matrizes fundiárias...")
    
    sucessos, pulados = 0, 0
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        tarefas = {executor.submit(processar_fazenda, index, row, ANO_IMAGEM): index 
                   for index, row in gdf_final.iterrows()}
        
        for futuro in as_completed(tarefas):
            try:
                resultados_fazenda = futuro.result()
                if resultados_fazenda:
                    for res in resultados_fazenda:
                        if res['status'] == 'pulado':
                            pulados += 1
                        elif res['status'] == 'sucesso':
                            sucessos += 1
                            adicionar_ao_csv_seguro(res['dados'], ficheiro_csv)
                            print(f"    [OK] {res['nome']} guardado! ({res['dados']['cultura_real']})")
                        elif res['status'] == 'erro':
                            print(f"    [ERRO] Falha no GEE para {res['nome']}.")
            except Exception as e:
                print(f"    [CRÍTICO] Erro ao processar futuro: {e}")

    print(f"\nResumo GEE do Ano {ANO_IMAGEM}: {sucessos} descarregados, {pulados} já existiam.")