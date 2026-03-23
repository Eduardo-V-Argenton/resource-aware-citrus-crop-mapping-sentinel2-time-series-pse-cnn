import pandas as pd
import numpy as np
import os

NUM_NULL = 0
FICHEIRO_CSV = 'dataset/dataset_index.csv'
PASTA_TENSORES = 'dataset/Tensores_Treino/'

df_csv = pd.read_csv(FICHEIRO_CSV)

files = os.listdir(PASTA_TENSORES)
files_base = [f[:-8] for f in files]
csv_base = df_csv['nome_base'].tolist()
set_files = set(files_base)
set_csv = set(csv_base)
so_files = set_files - set_csv
print("Só em files:", so_files)
so_csv = set_csv - set_files
print("Só no CSV:", so_csv)
invalid_file = []
null_images = []
all_null = []
more_12_null = []
len_total = len(df_csv)
for l,(_, linha) in enumerate(df_csv.iterrows()):
    print(f"{l}/{len_total} - Verificando {linha['nome_base']}...", end="\r")
    try:
        # t_10 = np.load(os.path.join(PASTA_TENSORES, f"{linha['nome_base']}_10m.npy"))
        t_20 = np.load(os.path.join(PASTA_TENSORES, f"{linha['nome_base']}_20m.npy"))
        is_all_null=True
        null_count = 0
        for i in range(0,24):
            if np.all(t_20[i, 1] == 0):
                null_images.append(f"{linha['nome_base']}_{i}")
                null_count += 1
            else:
                is_all_null=False
        if is_all_null:
            all_null.append(linha['nome_base'])
        if null_count > 12:
            more_12_null.append(linha['nome_base'])
    except Exception as e:
        invalid_file.append(linha['nome_base'])

print(f"\n-> Total de arquivos inválidos: {len(invalid_file)} ({len(invalid_file)/len_total:.2%})")
print(f"-> Total de imagens nulas: {len(null_images)} ({len(null_images)/(len_total*25):.2%})")
print(f"-> Total de bases com todas as imagens nulas: {len(all_null)} ({len(all_null)/len_total:.2%})")
print(f"-> Total de bases com mais de 12 imagens nulas: {len(more_12_null)} ({len(more_12_null)/len_total:.2%})")
df_filtrado = df_csv[df_csv['nome_base'].isin(more_12_null)]

count_por_ano = df_filtrado['ano'].value_counts().sort_index()

print(count_por_ano)

# apenas 2017 e 2018 tem muitos invalidos, logo podemos remove-los.

data = pd.read_csv("dataset/dataset_index.csv")
data = data[data["ano"] >= 2019]
len_total = len(data)
len_citrus = len(data[data["classe_mapbiomas"] == 47])
len_cana = len(data[data["classe_mapbiomas"] == 20])
len_cafe = len(data[data["classe_mapbiomas"] == 46])
len_pastagem = len(data[data["classe_mapbiomas"] == 15])
len_floresta = len(data[data["classe_mapbiomas"] == 3])
len_soja = len(data[data["classe_mapbiomas"] == 39])
len_sivicultura = len(data[data["classe_mapbiomas"] == 9])
len_false =  len_cana + len_cafe + len_pastagem + len_floresta + len_soja + len_sivicultura

print(f"Total de amostras: {len_total}")
print(f"Total de amostras de citros: {len_citrus} ({len_citrus/len_total:.2%})")
print(f"Total de amostras de cana: {len_cana} ({len_cana/len_total:.2%})")
print(f"Total de amostras de café: {len_cafe} ({len_cafe/len_total:.2%})")
print(f"Total de amostras de pastagem: {len_pastagem} ({len_pastagem/len_total:.2%})")
print(f"Total de amostras de floresta: {len_floresta} ({len_floresta/len_total:.2%})")
print(f"Total de amostras de soja: {len_soja} ({len_soja/len_total:.2%})")
print(f"Total de amostras de silvicultura: {len_sivicultura} ({len_sivicultura/len_total:.2%})")
print(f"Total de amostras false: {len_false/len_total:.2%}")