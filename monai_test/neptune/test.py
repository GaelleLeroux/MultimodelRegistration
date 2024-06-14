from PIL import Image
import numpy as np

def read_image_and_get_unique_pixel_values(image_path):
    # Ouvrir l'image
    image = Image.open(image_path)
    
    # Convertir l'image en tableau numpy
    image_array = np.array(image)
    
    # Obtenir les valeurs uniques des pixels
    unique_pixel_values = np.unique(image_array)
    
    return unique_pixel_values

# Chemin vers l'image PNG
image_path = '/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a1_MRI_Seg_2D/test/A002_MR_Seg_clean_slice_052.png'

unique_pixel_values = read_image_and_get_unique_pixel_values(image_path)

# Imprimer les valeurs uniques des pixels
print("Valeurs uniques des pixels :")
print(unique_pixel_values)