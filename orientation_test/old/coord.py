import nibabel as nib

def get_origin_and_orientation(nifti_file_path):
    # Charger le fichier NIfTI
    img = nib.load(nifti_file_path)
    
    # L'en-tête de l'image contient les informations métadonnées
    header = img.header
    
    # La matrice affine contient les informations sur l'origine et l'orientation
    affine = img.affine
    
    # L'origine est la dernière colonne de la matrice affine (sans la dernière ligne)
    origin = affine[:3, 3]
    
    # L'orientation peut être déduite des trois premières colonnes de la matrice affine
    orientation = affine[:3, :3]
    
    return origin, orientation

# Remplacer 'chemin/vers/votre/fichier.nii.gz' par le chemin réel vers votre fichier NIfTI
origin, orientation = get_origin_and_orientation('/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/imagesTr/M004_MRI.nii.gz')

print("Origin:", origin)
print("Orientation:\n", orientation)

print("*"*150)

import numpy as np

def print_nonzero_voxel_coordinates(nifti_file_path):
    # Charger le fichier NIfTI
    img = nib.load(nifti_file_path)
    
    # Accéder aux données de l'image sous forme d'un tableau numpy
    data = img.get_fdata()
    
    # Trouver les indices des voxels non nuls
    nonzero_voxel_indices = np.nonzero(data)
    
    # Convertir les indices en coordonnées spatiales
    # Ceci est fait en utilisant la transformation affine de l'image NIfTI
    # qui mappe les indices des voxels aux coordonnées dans un espace de référence
    affine = img.affine
    nonzero_voxel_coordinates = np.dot(affine[:3, :3], nonzero_voxel_indices) + affine[:3, 3][:, np.newaxis]
    
    # Imprimer les coordonnées
    for coord in nonzero_voxel_coordinates.T:  # Transposer pour itérer sur les colonnes
        print(coord)

# Remplacer 'chemin/vers/votre/fichier.nii' par le chemin vers votre fichier NIfTI
print_nonzero_voxel_coordinates('/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/imagesTr/M004_MRI.nii.gz')