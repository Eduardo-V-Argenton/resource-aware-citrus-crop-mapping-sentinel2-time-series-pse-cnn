import pandas as pd

data = pd.read_csv('/mnt/SSD_SATA/dataset/dataset_index.csv')
data["base_name"] = data["name"].str.replace(r"_p\d+", "", regex=True)

# Separate pieces and non-pieces
data_piece = data[data["is_piece"] == 1]
data_not_piece = data[data["is_piece"] == 0]

# Remove duplicates keeping only the first of each base_name
data_piece = data_piece.drop_duplicates(subset="base_name", keep="first")

# Join everything back together
data = pd.concat([data_not_piece, data_piece]).drop(columns=["base_name"])

total_len = data['area_ha'].sum()
citrus_len = data[data["mapbiomas_class"] == 47]['area_ha'].sum()
sugarcane_len = data[data["mapbiomas_class"] == 20]['area_ha'].sum()
coffee_len = data[data["mapbiomas_class"] == 46]['area_ha'].sum()
pasture_len = data[data["mapbiomas_class"] == 15]['area_ha'].sum()
forest_len = data[data["mapbiomas_class"] == 3]['area_ha'].sum()
soy_len = data[data["mapbiomas_class"] == 39]['area_ha'].sum()
silviculture_len = data[data["mapbiomas_class"] == 9]['area_ha'].sum()
flooded_len = data[data["mapbiomas_class"] == 11]['area_ha'].sum()

false_len = sugarcane_len + coffee_len + pasture_len + forest_len + soy_len + silviculture_len + flooded_len

print(f"Total samples (area): {total_len:.2f}")
print(f"Total citrus samples: {citrus_len:.2f} ({citrus_len/total_len:.2%})")
print(f"Total sugarcane samples: {sugarcane_len:.2f} ({sugarcane_len/total_len:.2%})")
print(f"Total coffee samples: {coffee_len:.2f} ({coffee_len/total_len:.2%})")
print(f"Total pasture samples: {pasture_len:.2f} ({pasture_len/total_len:.2%})")
print(f"Total forest samples: {forest_len:.2f} ({forest_len/total_len:.2%})")
print(f"Total soy samples: {soy_len:.2f} ({soy_len/total_len:.2%})")
print(f"Total silviculture samples: {silviculture_len:.2f} ({silviculture_len/total_len:.2%})")
print(f"Total flooded field samples: {flooded_len:.2f} ({flooded_len/total_len:.2%})")
print(f"Total false samples: {false_len/total_len:.2%}")
