import pandas as pd

data = pd.read_csv('/mnt/SSD_SATA/dataset/dataset_index.csv'
)
data["base_name"] = data["name"].str.replace(r"_p\d+", "", regex=True)

# Separar peças e não-peças
data_piece = data[data["is_piece"] == 1]
data_not_piece = data[data["is_piece"] == 0]

# Remover duplicados mantendo só o primeiro de cada base_name
data_piece = data_piece.drop_duplicates(subset="base_name", keep="first")

# Juntar tudo de volta
data = pd.concat([data_not_piece, data_piece]).drop(columns=["base_name"])

len_total = data['area_ha'].sum()
len_citrus = data[data["mapbiomas_class"] == 47]['area_ha'].sum()
len_cana = data[data["mapbiomas_class"] == 20]['area_ha'].sum()
len_cafe = data[data["mapbiomas_class"] == 46]['area_ha'].sum()
len_pastagem = data[data["mapbiomas_class"] == 15]['area_ha'].sum()
len_floresta = data[data["mapbiomas_class"] == 3]['area_ha'].sum()
len_soja = data[data["mapbiomas_class"] == 39]['area_ha'].sum()
len_sivicultura = data[data["mapbiomas_class"] == 9]['area_ha'].sum()
len_alagado = data[data["mapbiomas_class"] == 11]['area_ha'].sum()
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

