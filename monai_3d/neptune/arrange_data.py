import os
import shutil

def copier_fichiers_crops(source_dir, destination_dir):
    # Vérifier si le dossier de destination existe, sinon le créer
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Parcourir le dossier source
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if "CBCT_Seg" in file:
                # print(1)
                source_file = os.path.join(root, file)
                destination_file = os.path.join(destination_dir, file)
                
                # Copier le fichier dans le dossier de destination
                shutil.copy2(source_file, destination_file)
                print(f"Fichier copié: {source_file} vers {destination_file}")

# Exemple d'utilisation
source_dir = "/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/3D_more_data/12_Leroux_Action_ready_(1)/12_Leroux_Action_ready"
destination_dir = "/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/3D_more_data/a0_CBCT_seg"
copier_fichiers_crops(source_dir, destination_dir)
