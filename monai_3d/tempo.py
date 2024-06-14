import os
import shutil

def move_files_with_MR(input_dir, output_dir):
    # Vérifier si le dossier de sortie existe, sinon le créer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parcourir tous les fichiers dans le dossier d'entrée
    for filename in os.listdir(input_dir):
        # Vérifier si "MR" est dans le nom du fichier
        if "MR" in filename:
            # Construire les chemins complets source et destination
            src_path = os.path.join(input_dir, filename)
            dest_path = os.path.join(output_dir, filename)
            # Déplacer le fichier
            shutil.move(src_path, dest_path)
            print(f"Moved: {src_path} -> {dest_path}")

# Chemins des dossiers d'entrée et de sortie
input_directory = '/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/3D/MRI-CB'
output_directory = '/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/3D/CB_a01_MRI/train'

# Appel de la fonction pour déplacer les fichiers
move_files_with_MR(input_directory, output_directory)
