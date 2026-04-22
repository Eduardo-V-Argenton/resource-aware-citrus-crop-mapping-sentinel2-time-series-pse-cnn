import pandas as pd

data = pd.read_csv('/mnt/SSD_SATA/dataset/dataset_index.csv'
)
len_total = len(data)
len_citrus = len(data[data["mapbiomas_class"] == 47])
len_cana = len(data[data["mapbiomas_class"] == 20])
len_cafe = len(data[data["mapbiomas_class"] == 46])
len_pastagem = len(data[data["mapbiomas_class"] == 15])
len_floresta = len(data[data["mapbiomas_class"] == 3])
len_soja = len(data[data["mapbiomas_class"] == 39])
len_sivicultura = len(data[data["mapbiomas_class"] == 9])
len_alagado = len(data[data["mapbiomas_class"] == 11])
len_false =  len_cana + len_cafe + len_pastagem + len_floresta + len_soja + len_sivicultura + len_alagado

print(f"Total de amostras: {len_total}")
print(f"Total de amostras de citros: {len_citrus} ({len_citrus/len_total:.2%})")
print(f"Total de amostras de cana: {len_cana} ({len_cana/len_total:.2%})")
print(f"Total de amostras de café: {len_cafe} ({len_cafe/len_total:.2%})")
print(f"Total de amostras de pastagem: {len_pastagem} ({len_pastagem/len_total:.2%})")
print(f"Total de amostras de floresta: {len_floresta} ({len_floresta/len_total:.2%})")
print(f"Total de amostras de soja: {len_soja} ({len_soja/len_total:.2%})")
print(f"Total de amostras de silvicultura: {len_sivicultura} ({len_sivicultura/len_total:.2%})")
print(f"Total de amostras de campo alagado: {len_alagado} ({len_alagado/len_total:.2%})")
print(f"Total de amostras false: {len_false/len_total:.2%}")

