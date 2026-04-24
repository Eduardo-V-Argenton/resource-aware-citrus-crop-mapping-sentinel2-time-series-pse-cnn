import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", r"Mean of empty slice")

# =====================================================================
# DIRECTORY CONFIGURATIONS AND PARAMETERS
# =====================================================================
CSV_FILE = '/mnt/SSD_SATA/dataset/dataset_index.csv'
SOURCE_FOLDER = "/mnt/SSD_SATA/dataset/Tensores_Treino/"

# =====================================================================
# HELPER FUNCTION: BASE PROCESSING
# =====================================================================
def verify_small(base_name):
    """Loads bands, performs upsampling, clipping, and gap-filling. Returns tensor with NaNs in the background."""
    t_10 = (
        np.load(os.path.join(SOURCE_FOLDER, f"{base_name}_10m.npy")).astype(np.float32)
        / 10000.0
    )

    _, _, h_10, w_10 = t_10.shape

    # Extraction and Stacking
    b2, b3, b4, b8 = t_10[:, 0], t_10[:, 1], t_10[:, 2], t_10[:, 3]
    base_tensor = np.stack((b2, b3, b4, b8), axis=1)

    # Cleaning and Clipping (-2.0 to 2.0)
    base_tensor = np.nan_to_num(base_tensor, nan=np.nan, posinf=2.0, neginf=-2.0)
    base_tensor = np.clip(base_tensor, a_min=-2.0, a_max=2.0)
    
    farm_mask = (base_tensor[0, 0, :, :] != 0.0) & (base_tensor[0, 0, :, :] > -0.5)
    flat_base = base_tensor[:, :, farm_mask]
    t, c, p = flat_base.shape
    if p < 32:
        return False
    return True

# =====================================================================
# PROCESSING AND SAVING (No Normalization)
# =====================================================================

data = pd.read_csv('/mnt/SSD_SATA/dataset/dataset_index.csv')
data_len = len(data)
small_counter = 0

for i, row in enumerate(data.itertuples(index=False)):
    print(f"{i}/{data_len}", end="\r")
    if not verify_small(row.name):
        data.drop(data[data['name'] == row.name].index, inplace=True)
        small_counter += 1

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
print(f"Total samples removed for being too small: {small_counter} ({small_counter/data_len:.2%})")
