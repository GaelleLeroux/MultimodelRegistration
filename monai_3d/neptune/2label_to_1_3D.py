import os
import nibabel as nib
import numpy as np

# Définissez les chemins vers les dossiers
input_folder = '/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/3D_more_data/a0_MRI_seg'
output_folder_base = '/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/3D_more_data/a1_MRI_seg'

# Créez le dossier de base pour les sorties s'il n'existe pas
os.makedirs(output_folder_base, exist_ok=True)

# Fonction pour séparer et sauvegarder les segmentations
def save_segmentations(file_path, output_folder_base):
    # Chargez le fichier MRI
    img = nib.load(file_path)
    data = img.get_fdata()

    # Trouvez tous les labels uniques présents dans les données
    unique_labels = np.unique(data)

    # Parcourez chaque label unique et sauvegardez la segmentation correspondante
    for label in unique_labels:
        if label == 0:  # Ignorer l'arrière-plan si label 0
            continue

        # Séparez la segmentation pour le label courant
        label_data = (data == label).astype(np.uint8)

        # Définissez le nouveau nom de fichier
        file_name = os.path.basename(file_path).replace('.nii.gz', f'_label{int(label)}.nii.gz')
        output_folder_label = os.path.join(output_folder_base, f'label_{int(label)}')
        os.makedirs(output_folder_label, exist_ok=True)
        file_name_label = os.path.join(output_folder_label, file_name)

        # Créez la nouvelle image Nifti
        img_label = nib.Nifti1Image(label_data, img.affine, img.header)

        # Sauvegardez la nouvelle image
        nib.save(img_label, file_name_label)

# Parcourez tous les fichiers dans le dossier d'entrée
for file_name in os.listdir(input_folder):
    # if file_name.endswith('MRI_seg.nii.gz'):
        file_path = os.path.join(input_folder, file_name)
        save_segmentations(file_path, output_folder_base)

print("Segmentation des fichiers terminée.")
