import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


# Charger le fichier NIfTI
# nifti_file = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/volumes/M002_CBCT.nii.gz'
nifti_file = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/volumes/test_transform/M002_transforme_CBCT.nii.gz"
img = nib.load(nifti_file)

# Obtenir l'en-tête du fichier NIfTI
header = img.header

# Afficher les informations pouvant correspondre à celles du DICOM
# Note : Ces informations sont inférées et peuvent ne pas correspondre exactement aux métadonnées DICOM originales
print(f"Dimensiosn de l'image: {header.get_data_shape()}")
print(f"Taille des voxels (mm): {header.get_zooms()}")

# La matrice d'affinage peut être utilisée pour inférer la position et l'orientation
affine_matrix = img.affine
print(f"Matrice d'affinage: \n{affine_matrix}")

# Les informations exactes comme `ImagePositionPatient`, `ImageOrientationPatient`, etc., ne sont pas directement disponibles


# # Charger le fichier NIfTI
# nifti_file_path = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/volumes/M002_CBCT.nii.gz'
# nifti_img = nib.load(nifti_file_path)

# # Définir la matrice de transformation
# transformation_matrix = np.array([
#     [1/0.3, 0, 0, -115.2/0.3],
#     [0, 1/0.3, 0, -115.2/0.3],
#     [0, 0, 1/0.3, 51.62/0.3],
#     [0, 0, 0, 1]
# ])

# # Appliquer la transformation à la matrice d'affinage
# new_affine = np.dot(transformation_matrix, nifti_img.affine)

# # Créer une nouvelle image NIfTI avec l'affine transformée
# transformed_img = nib.Nifti1Image(nifti_img.get_fdata(), affine=new_affine)

# # Sauvegarder l'image transformée
# output_file_path = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/volumes/test_transform/M002_transforme_CBCT.nii.gz" # Modifiez ceci avec le chemin où vous souhaitez sauvegarder
# nib.save(transformed_img, output_file_path)

# print("Transformation appliquée et fichier sauvegardé.")
def lire_fichier_nifti(chemin_fichier):
    """Lire un fichier NIfTI et retourner l'objet image et sa matrice d'affinage."""
    img = nib.load(chemin_fichier)
    return img, img.affine

def creer_matrice_transformation(affine):
    """Créer une matrice de transformation pour centrer et aligner le volume NIfTI."""
    # Extraire la taille des voxels
    voxel_dims = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    # Calculer le centre du volume en coordonnées voxel
    centre_voxel = np.linalg.inv(affine) @ np.array([0, 0, 0, 1])

    # Créer une matrice de translation pour centrer le volume
    translation = np.eye(4)
    translation[:3, 3] = -centre_voxel[:3] * voxel_dims

    # Créer une matrice de mise à l'échelle pour normaliser les dimensions des voxels
    scaling = np.eye(4)
    scaling[0, 0] = 1 / voxel_dims[0]
    scaling[1, 1] = 1 / voxel_dims[1]
    scaling[2, 2] = 1 / voxel_dims[2]

    # La matrice de transformation est le produit de la mise à l'échelle et de la translation
    transformation_matrix = scaling @ translation
    return transformation_matrix

def appliquer_transformation_et_sauvegarder(img, nouvelle_affine, chemin_sortie):
    """Appliquer la matrice de transformation à l'image NIfTI et sauvegarder le résultat."""
    img_transformee = nib.Nifti1Image(img.get_fdata(), affine=nouvelle_affine)
    nib.save(img_transformee, chemin_sortie)


# # Utilisation des fonctions
# chemin_fichier = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/volumes/M002_MRI.nii.gz'
# chemin_sortie = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/volumes/test_transform/M002_transorm2_MRI.nii.gz'

# # Lire le fichier NIfTI
# img, affine = lire_fichier_nifti(chemin_fichier)
# print("affine : \n",affine)

# # Créer la matrice de transformation
# transformation_matrix = creer_matrice_transformation(affine)
# print("transformation_matrix : \n",transformation_matrix)

# # Calculer la nouvelle matrice d'affinage
# nouvelle_affine = np.dot(transformation_matrix, affine)

# # Appliquer la transformation et sauvegarder le résultat
# appliquer_transformation_et_sauvegarder(img, nouvelle_affine, chemin_sortie)

print("Le fichier NIfTI transformé a été sauvegardé.")
def lire_fichier_nifti2(chemin_fichier):
    """Lire un fichier NIfTI et retourner les données et la matrice d'affinage."""
    nii = nib.load(chemin_fichier)
    donnees = nii.get_fdata()
    affine = nii.affine
    return donnees, affine

def creer_matrice_transformation(affine, dimensions_voxels):
    """Créer une matrice de transformation pour centrer et aligner le volume."""
    dimensions_voxels_array = np.array(dimensions_voxels)  # Convertir en array NumPy
    centre = np.dot(affine, np.append(dimensions_voxels_array / 2, 1))
    translation = np.eye(4)
    translation[:3, 3] = -centre[:3]
    return translation


def appliquer_transformation(donnees, affine, transformation):
    """Appliquer la transformation aux données en utilisant l'interpolation."""
    # Calculer la nouvelle matrice d'affinage
    nouvelle_affine = np.dot(transformation, affine)
    # Appliquer la transformation
    donnees_transformees = affine_transform(donnees, nouvelle_affine[:3, :3], offset=nouvelle_affine[:3, 3], order=1)
    return donnees_transformees, nouvelle_affine

# Utilisation des fonctions
chemin_fichier = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/volumes/M002_MRI.nii.gz"
donnees, affine = lire_fichier_nifti2(chemin_fichier)
dimensions_voxels = donnees.shape
print("affine : ",affine)
print("dimensions_voxels : ",dimensions_voxels)
# print("donnees : ",donnees)

transformation = creer_matrice_transformation(affine, dimensions_voxels)
donnees_transformees, nouvelle_affine = appliquer_transformation(donnees, affine, transformation)

# Pour sauvegarder le résultat
nii_transforme = nib.Nifti1Image(donnees_transformees, nouvelle_affine)
nib.save(nii_transforme, "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/volumes/test_transform/M002_transorm4_MRI.nii.gz")