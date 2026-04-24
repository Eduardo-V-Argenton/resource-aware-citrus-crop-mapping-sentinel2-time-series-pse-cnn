import pandas as pd

data = pd.read_csv('/mnt/SSD_SATA/dataset/dataset_index.csv')

total_len = len(data)
citrus_len = len(data[data["mapbiomas_class"] == 47])
sugarcane_len = len(data[data["mapbiomas_class"] == 20])
coffee_len = len(data[data["mapbiomas_class"] == 46])
pasture_len = len(data[data["mapbiomas_class"] == 15])
forest_len = len(data[data["mapbiomas_class"] == 3])
soy_len = len(data[data["mapbiomas_class"] == 39])
silviculture_len = len(data[data["mapbiomas_class"] == 9])
flooded_len = len(data[data["mapbiomas_class"] == 11])

false_len = sugarcane_len + coffee_len + pasture_len + forest_len + soy_len + silviculture_len + flooded_len

print(f"Total samples: {total_len}")
print(f"Total citrus samples: {citrus_len} ({citrus_len/total_len:.2%})")
print(f"Total sugarcane samples: {sugarcane_len} ({sugarcane_len/total_len:.2%})")
print(f"Total coffee samples: {coffee_len} ({coffee_len/total_len:.2%})")
print(f"Total pasture samples: {pasture_len} ({pasture_len/total_len:.2%})")
print(f"Total forest samples: {forest_len} ({forest_len/total_len:.2%})")
print(f"Total soy samples: {soy_len} ({soy_len/total_len:.2%})")
print(f"Total silviculture samples: {silviculture_len} ({silviculture_len/total_len:.2%})")
print(f"Total flooded field samples: {flooded_len} ({flooded_len/total_len:.2%})")
print(f"Total false samples: {false_len/total_len:.2%}")
