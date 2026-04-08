import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

FICHEIRO_CSV = 'dataset/dataset_index.csv'
PASTA_ORIGEM = '/mnt/SSD_SATA/Tensores_Treino/'

# =====================================================================
# FUNÇÃO DE ANÁLISE (Conta as nuvens sem gravar nada)
# =====================================================================
def analisar_poligono(nome_base):
    try:
        # 1. Carrega os tensores
        caminho_10m = os.path.join(PASTA_ORIGEM, f"{nome_base}_10m.npy")
        caminho_20m = os.path.join(PASTA_ORIGEM, f"{nome_base}_20m.npy")
        
        if not os.path.exists(caminho_10m) or not os.path.exists(caminho_20m):
            return None

        t_10 = np.load(caminho_10m).astype(np.float32) / 10000.0
        t_20 = np.load(caminho_20m).astype(np.float32) / 10000.0
        
        _, _, h_10, w_10 = t_10.shape
        if h_10 < 15 or w_10 < 15:
            return None
            
        # 2. ZOOM
        tensor_20_tmp = torch.from_numpy(t_20)
        t_20_zoom = F.interpolate(tensor_20_tmp, size=(h_10, w_10), mode='bilinear', align_corners=False).numpy()
        
        # 3. Extrai Bandas para a Máscara
        b2 = t_10[:, 0, :, :]
        b3 = t_10[:, 1, :, :]
        b4 = t_10[:, 2, :, :]
        b8 = t_10[:, 3, :, :]
        
        # 4. Aplica exatamente as mesmas regras do seu script principal
        nodata_mask = (b2 + b3 + b4 + b8) == 0
        cloud_mask = b2 > 0.15
        shadow_mask = b8 < 0.08
        mascara_destruicao = nodata_mask | cloud_mask | shadow_mask
        
        # Em vez de calcular índices pesados, usamos uma matriz de uns para testar os NaNs
        tensor_teste = np.ones_like(b2)
        tensor_teste[mascara_destruicao] = np.nan
        
        # 5. Cálculo das estatísticas deste polígono
        total_pixels = tensor_teste.size
        nans_count = np.isnan(tensor_teste).sum()
        taxa_nans = nans_count / total_pixels
        
        pixel_morto = bool(np.isnan(tensor_teste).all(axis=0).any())
        
        return {
            'nome_base': nome_base, 
            'taxa_nans': taxa_nans, 
            'pixel_morto': pixel_morto
        }
        
    except Exception:
        return None

# =====================================================================
# MOTOR PRINCIPAL E RELATÓRIO
# =====================================================================
if __name__ == '__main__':
    df_csv = pd.read_csv(FICHEIRO_CSV)
    df_csv = df_csv[df_csv['ano'] >= 2019]
    nomes_bases = df_csv['nome_base'].tolist()

    resultados = []
    nucleos = max(1, os.cpu_count() - 1)

    print(f"A iniciar análise de nuvens em {len(nomes_bases)} polígonos...")
    
    with ProcessPoolExecutor(max_workers=nucleos) as executor:
        futuros = {executor.submit(analisar_poligono, nome): nome for nome in nomes_bases}
        
        for futuro in tqdm(as_completed(futuros), total=len(futuros), desc="Varrendo NaNs"):
            res = futuro.result()
            if res is not None:
                resultados.append(res)
                
    # Cria um DataFrame com os resultados para calcular as estatísticas facilmente
    df_res = pd.DataFrame(resultados)
    
    print("\n" + "="*60)
    print(" RELATÓRIO DE DESTRUIÇÃO ATMOSFÉRICA (NaNs)")
    print("="*60)
    
    total_validos = len(df_res)
    mortos = df_res['pixel_morto'].sum()
    print(f"Total de polígonos lidos com sucesso: {total_validos}")
    print(f"Polígonos descartados por terem 'Pixels 100% Cegos': {mortos} ({(mortos/total_validos)*100:.1f}%)")
    
    print("\n--- Distribuição Real da Taxa de NaNs ---")
    taxas = df_res['taxa_nans']
    print(f"Média Geral de Nuvens: {taxas.mean():.1%}")
    print(f"Mediana (50% do dataset tem MENOS de): {taxas.median():.1%}")
    print(f"Percentil 75% (75% do dataset tem MENOS de): {taxas.quantile(0.75):.1%}")
    print(f"Percentil 85% (85% do dataset tem MENOS de): {taxas.quantile(0.85):.1%}")
    print(f"Percentil 95% (95% do dataset tem MENOS de): {taxas.quantile(0.95):.1%}")
    
    print("\n--- SIMULAÇÃO DE LIMITES PARA O SEU MODELO ---")
    print("Quantas fazendas sobram se você definir o LIMITE_MAXIMO_NANS para:")
    
    limit_tests = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    for limite in limit_tests:
        # Aprova se estiver abaixo do limite E NÃO tiver pixels cegos
        aprovados = len(df_res[(df_res['taxa_nans'] <= limite) & (~df_res['pixel_morto'])])
        perdidos = total_validos - aprovados
        pct_perdidos = (perdidos/total_validos)*100
        
        print(f" > Limite {limite:.2f} ({int(limite*100)}%): Sobram {aprovados} fazendas | Perde {perdidos} ({pct_perdidos:.1f}%)")
        
    print("="*60)