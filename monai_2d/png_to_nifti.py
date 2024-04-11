import os
import numpy as np
import nibabel as nib
from PIL import Image
import argparse

def pngs_to_nifti(args):
    png_folder = args.input_PNG
    mri_file = args.input_MRI
    output_file = args.output
    
    # Charger le fichier MRI d'entrée
    mri_img = nib.load(mri_file)
    mri_data = mri_img.get_fdata()
    
    # S'assurer que mri_data est mutable
    mri_data = np.copy(mri_data)
    
    # Lister tous les fichiers PNG dans le dossier
    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')])
    
    # Lire la première image pour obtenir les dimensions (supposé identiques pour toutes les images PNG)
    example_img = Image.open(os.path.join(png_folder, png_files[0]))
    img_array = np.array(example_img)
    
    # S'assurer que la forme des données MRI est compatible avec les images PNG
    assert mri_data.shape[0] == img_array.shape[0] and mri_data.shape[1] == img_array.shape[1], "Les dimensions du MRI et des PNG ne correspondent pas."
    
    # Remettre à zéro les valeurs sur les axes x et y si nécessaire (dépend de la logique spécifique requise)
    # mri_data[:, :, :] = 0 # Décommenter si vous voulez vraiment remettre à zéro toute la matrice
    
    # Lire chaque image PNG et la stocker dans le tableau des données MRI
    for i, png_file in enumerate(png_files):
        img_path = os.path.join(png_folder, png_file)
        img = Image.open(img_path)
        mri_data[:, :, i] = np.array(img)  # Assurez-vous que cette opération est logique par rapport à la structure des données MRI
    
    # Créer un nouvel objet NIfTI avec les données mises à jour
    new_mri_img = nib.Nifti1Image(mri_data, affine=mri_img.affine)
    
    # Sauvegarder l'objet NIfTI en fichier .nii.gz
    nib.save(new_mri_img, output_file)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Convert PNGs to NIfTI and integrate with MRI data')
    parser.add_argument('--input_PNG', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/2D_Training/LabelsTr/B001_MRI/', help='PNG folder to put in nifti')
    parser.add_argument('--input_MRI', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/2D_Training/zz01_reconstruct_from_png/B001_MRI.nii.gz', help='Input MRI file to be integrated with PNG data')
    parser.add_argument('--output', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/2D_Training/zz01_reconstruct_from_png/B001_MRI_label_reconstruct.nii.gz', help='path output for the updated nifti')
    args = parser.parse_args()

    pngs_to_nifti(args)
