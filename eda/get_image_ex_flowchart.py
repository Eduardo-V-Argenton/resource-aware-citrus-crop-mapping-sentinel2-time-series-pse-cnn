import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn.functional as F
import os

SOURCE_FOLDER = "/mnt/SSD_SATA/dataset/Tensores_Treino/"
BASE_NAME = "/mnt/SSD_SATA/dataset/Tensores_Treino/img_00321_2020" 
TIME_STEP = 12 

BAND_NAMES = ['B2', 'B3', 'B4', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']

BAND_CMAPS = [
    'Blues', 'Greens', 'Reds', 'Purples', 
    'Oranges', 'OrRd', 'YlOrRd', 'PuRd', 
    'YlOrBr', 'copper'
]

plt.rcParams["font.family"] = "serif"

t_10 = np.load(os.path.join(SOURCE_FOLDER, f"{BASE_NAME}_10m.npy")).astype(np.float32) / 10000.0
t_20 = np.load(os.path.join(SOURCE_FOLDER, f"{BASE_NAME}_20m.npy")).astype(np.float32) / 10000.0

_, _, h_10, w_10 = t_10.shape
t_20_up = F.interpolate(torch.from_numpy(t_20), size=(h_10, w_10), mode="nearest").numpy()

b2, b3, b4, b8 = t_10[:, 0], t_10[:, 1], t_10[:, 2], t_10[:, 3]
b5, b6, b7, b8a, b11, b12 = t_20_up[:, 0], t_20_up[:, 1], t_20_up[:, 2], t_20_up[:, 3], t_20_up[:, 4], t_20_up[:, 5]

tensor_4d = np.stack((b2, b3, b4, b8, b5, b6, b7, b8a, b11, b12), axis=1)

tensor_4d = np.where(tensor_4d == 0.0, np.nan, tensor_4d)
image_data = tensor_4d[TIME_STEP]

# ==========================================
# 3. PLOTAGEM 3D EMPILHADA (O Efeito "Álbum")
# ==========================================
fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(111, projection='3d')

H, W = image_data.shape[1], image_data.shape[2]
X, Y = np.meshgrid(np.arange(W), np.arange(H))

Z_STEP = 10 

for i in range(9, -1, -1):
    band_array = image_data[i].copy()
    
    valid_mask = (~np.isnan(band_array)) & (band_array > 0.001)
    valid_pixels = band_array[valid_mask]
    
    if len(valid_pixels) > 10:
        vmin = np.percentile(valid_pixels, 1)
        vmax = np.percentile(valid_pixels, 99)
    else:
        vmin, vmax = 0.0, 1.0
        
    Z = np.full((H, W), i * Z_STEP)
    
    cmap = cm.get_cmap(BAND_CMAPS[i]).copy()
    cmap.set_bad(color='white', alpha=0.0)
    
    surf = ax.plot_surface(X, Y, Z, facecolors=cmap((band_array - vmin) / (vmax - vmin)), 
                           shade=False, alpha=0.85, rstride=1, cstride=1, linewidth=0)
    
ax.view_init(elev=25, azim=-60)

ax.axis('off')
ax.grid(False)
ax.set_zlim(0, 10 * Z_STEP)

plt.tight_layout()
output_name = "image_ex_flowchart"
plt.savefig(output_name, dpi=300, bbox_inches='tight', transparent=True)
print(f"Cubo 3D gerado com sucesso: {output_name}")

plt.show()