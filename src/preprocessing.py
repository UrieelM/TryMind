#region Imports and Setup
import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
#endregion

#region Image Loading and Preprocessing
# Image path to analyze
img_path = "dataset/000000007459.jpg" 

# Read image and convert to grayscale
img = cv2.imread(img_path)
if img is None:
    raise ValueError("Could not load the image.")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#endregion

#region CLAHE Application
# CLAHE parameters
clip_limit = 5  # Contrast limit for adaptive histogram equalization
tile_grid_size = (8, 8)  # Size of grid for histogram equalization

def clip_mapping(clip_limit, win_size):
    area = win_size[0] * win_size[1]
    T_min = 256 / area
    T_max = 256
    mapped_clip = min(max((clip_limit * 256) / area, T_min), T_max)
    return mapped_clip

# Compute mapped clip limit and create CLAHE object
mapped_clip = clip_mapping(clip_limit, tile_grid_size)
print(f"Clip limit Tu: {clip_limit}")
print(f"Clip limit mapeado (Tc): {mapped_clip}")

clahe = cv2.createCLAHE(clipLimit=mapped_clip, tileGridSize=tile_grid_size)
# Apply CLAHE to the grayscale image
img_clahe = clahe.apply(gray)

# Intenta imprimir el valor del objeto CLAHE si está disponible
try:
    actual_clip = getattr(clahe, 'clipLimit', None)
    if actual_clip is None and hasattr(clahe, 'getClipLimit'):
        actual_clip = clahe.getClipLimit()
    print(f"Clip limit desde el objeto CLAHE: {actual_clip}")
except Exception as e:
    print(f"No se pudo leer clip limit desde el objeto CLAHE: {e}")
# Mostrar imágenes usando matplotlib en una ventana
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title("Imagen original")
axes[0].axis('off')

axes[1].imshow(img_clahe, cmap='gray')
axes[1].set_title("Imagen CLAHE")
axes[1].axis('off')

plt.tight_layout()
plt.show()