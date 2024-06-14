import argparse
import os
from PIL import Image
import numpy as np

def modify_image_pixel_values(image_path, output_folder):
    # Ouvrir l'image
    image = Image.open(image_path)
    
    # Convertir l'image en tableau numpy
    image_array = np.array(image)
    
    # Modifier les valeurs des pixels
    
    image_array[image_array == 127] = 0
    image_array[image_array == 255] = 255
    image_array[image_array == 0] = 0
    
    if np.all(image_array == 0):
        print(f"Image {os.path.basename(image_path)} est entièrement noire et ne sera pas enregistrée.")
        return
    
    # Convertir le tableau numpy modifié en image
    modified_image = Image.fromarray(image_array)
    
    # Créer le chemin de sortie
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, image_name)
    
    # Enregistrer l'image modifiée
    modified_image.save(output_path)
    print(f"Image modifiée enregistrée à : {output_path}")

def process_directory(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):
            image_path = os.path.join(input_folder, file_name)
            modify_image_pixel_values(image_path, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modifier les valeurs de pixels des images PNG.")
    parser.add_argument("--in_dir", type=str, help="Répertoire contenant les images PNG à traiter.")
    parser.add_argument("--out_dir", type=str, help="Répertoire où enregistrer les images modifiées.")
    
    args = parser.parse_args()
    
    process_directory(args.in_dir, args.out_dir)
