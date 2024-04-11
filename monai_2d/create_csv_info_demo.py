from PIL import Image
import os
import csv

# Chemin vers le dossier contenant les images .png
images_dir = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/2D_Training/demo/'

# Chemin vers le fichier .csv de sortie
csv_file_path = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/2D_Training/demo_info_size.csv'

# Ouvrir (ou créer) le fichier .csv en mode écriture
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Écrire l'en-tête du fichier .csv
    writer.writerow(['Nom du fichier', 'Largeur', 'Hauteur'])
    
    # Itérer sur tous les fichiers du dossier spécifié
    for file_name in os.listdir(images_dir):
        # Construire le chemin complet vers l'image
        file_path = os.path.join(images_dir, file_name)
        
        # Vérifier si le fichier est un .png
        if file_name.endswith('.png'):
            # Ouvrir l'image et lire ses dimensions
            with Image.open(file_path) as img:
                width, height = img.size
                
                # Écrire le nom du fichier et ses dimensions dans le fichier .csv
                writer.writerow([file_name, width, height])

print(f'Le fichier .csv a été créé avec succès : {csv_file_path}')