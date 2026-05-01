import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn.functional as F
import os

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
SOURCE_FOLDER = "/mnt/SSD_SATA/dataset/Tensores_Treino/"
BASE_NAME = "img_00321_2020" 

BAND_NAMES = [
    'B2 (Blue)', 'B3 (Green)', 'B4 (Red)', 'B8 (NIR)', 
    'B5 (Red Edge 1)', 'B6 (Red Edge 2)', 'B7 (Red Edge 3)', 
    'B8A (Narrow NIR)', 'B11 (SWIR 1)', 'B12 (SWIR 2)'
]

plt.rcParams["font.family"] = "serif"

# ==========================================
# 2. CARREGAMENTO E RECONSTRUÇÃO 4D
# ==========================================
print(f"Reconstruindo tensor 4D para: {BASE_NAME}")

t_10 = np.load(os.path.join(SOURCE_FOLDER, f"{BASE_NAME}_10m.npy")).astype(np.float32) / 10000.0
t_20 = np.load(os.path.join(SOURCE_FOLDER, f"{BASE_NAME}_20m.npy")).astype(np.float32) / 10000.0

_, _, h_10, w_10 = t_10.shape

t_20_tensor = torch.from_numpy(t_20)
t_20_up = F.interpolate(t_20_tensor, size=(h_10, w_10), mode="nearest").numpy()

b2, b3, b4, b8 = t_10[:, 0], t_10[:, 1], t_10[:, 2], t_10[:, 3]
b5, b6, b7, b8a, b11, b12 = t_20_up[:, 0], t_20_up[:, 1], t_20_up[:, 2], t_20_up[:, 3], t_20_up[:, 4], t_20_up[:, 5]

tensor_4d = np.stack((b2, b3, b4, b8, b5, b6, b7, b8a, b11, b12), axis=1)

# TRUQUE DE MESTRE: Substitui os Zeros (fundo) por NaN para o fundo ficar invisível
tensor_4d = np.where(tensor_4d == 0.0, np.nan, tensor_4d)

TIME_STEP = 12 
image_data = tensor_4d[TIME_STEP] 

# ==========================================
# 3. PLOTAGEM COM CORES SEMÂNTICAS
# ==========================================
fig, axes = plt.subplots(3, 4, figsize=(22, 12))
axes = axes.flatten()

BAND_CMAPS = [
    'Blues', 'Greens', 'Reds', 'Purples', 
    'Oranges', 'OrRd', 'YlOrRd', 'PuRd', 
    'YlOrBr', 'copper'
]

for i in range(10):
    ax = axes[i]
    band_array = image_data[i].copy() 
    
    valid_mask = (~np.isnan(band_array)) & (band_array > 0.001)
    valid_pixels = band_array[valid_mask]
    
    if len(valid_pixels) > 10:
        vmin = np.percentile(valid_pixels, 1)
        vmax = np.percentile(valid_pixels, 99)
    else:
        vmin, vmax = 0.0, 1.0
        
    print(f"[{BAND_NAMES[i]}] Escala calculada: Min {vmin:.3f} | Max {vmax:.3f}")
        
    band_array[~valid_mask] = np.nan
    
    cmap = cm.get_cmap(BAND_CMAPS[i]).copy()
    cmap.set_bad(color='white')
    
    im = ax.imshow(band_array, cmap=cmap, vmin=vmin, vmax=vmax)
    
    H, W = band_array.shape
    pad_y = H * 0.15  
    pad_x = W * 0.15  
    
    ax.set_xlim(-pad_x, W + pad_x)
    ax.set_ylim(H + pad_y, -pad_y) 
    # ---------------------------

    ax.set_title(BAND_NAMES[i], fontsize=16, fontweight='bold', pad=10)
    ax.axis('off')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=10)
for j in range(10, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
output_name = "images_ex.png"
plt.savefig(output_name, dpi=300, bbox_inches='tight', transparent=False)
print(f"Imagem salva com sucesso: {output_name}")

plt.show()