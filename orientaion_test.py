import pydicom
import numpy as np
import os
from scipy.ndimage import affine_transform

from orientation import orientation

def get_coordinates(dicom_file):
    # Charger le fichier DICOM
    ds = pydicom.dcmread(dicom_file)
    # Extraire les données de l'image sous forme d'un tableau numpy
    image = ds.pixel_array

    # Calculer les coordonnées du centre
    centre = (image.shape[0] // 2, image.shape[1] // 2)

    # Trouver le point le plus à droite au centre
    # C'est simplement le point le plus à droite sur la ligne centrale
    plus_a_droite_au_centre = (centre[0], image.shape[1] - 1)

    # Trouver le point le plus en haut au centre
    # C'est simplement le point le plus en haut sur la colonne centrale
    plus_en_haut_au_centre = (0, centre[1])

    return centre, plus_a_droite_au_centre, plus_en_haut_au_centre

def get_3d_coordinates(dicom_file):
    # Charger le fichier DICOM
    ds = pydicom.dcmread(dicom_file)
    # Extraire les données de l'image sous forme d'un tableau numpy
    image = ds.pixel_array

    # Obtenir l'orientation et la position de l'image dans l'espace patient
    image_position = np.array(ds.ImagePositionPatient)
    image_orientation = np.array(ds.ImageOrientationPatient).reshape(2, 3)
    row_orientation, col_orientation = image_orientation

    # Taille du pixel
    pixel_spacing = np.array(ds.PixelSpacing)

    # Calculer les coordonnées du centre
    centre_idx = np.array([image.shape[0] // 2, image.shape[1] // 2])
    centre_3d = image_position + centre_idx[0] * pixel_spacing[0] * row_orientation + centre_idx[1] * pixel_spacing[1] * col_orientation

    # Calculer le point le plus à droite au centre
    plus_a_droite_au_centre_idx = np.array([centre_idx[0], image.shape[1] - 1])
    plus_a_droite_au_centre_3d = image_position + plus_a_droite_au_centre_idx[0] * pixel_spacing[0] * row_orientation + plus_a_droite_au_centre_idx[1] * pixel_spacing[1] * col_orientation

    # Calculer le point le plus en haut au centre
    plus_en_haut_au_centre_idx = np.array([0, centre_idx[1]])
    plus_en_haut_au_centre_3d = image_position + plus_en_haut_au_centre_idx[0] * pixel_spacing[0] * row_orientation + plus_en_haut_au_centre_idx[1] * pixel_spacing[1] * col_orientation

    return centre_3d, plus_a_droite_au_centre_3d, plus_en_haut_au_centre_3d

# Remplacer 'chemin_vers_votre_fichier.dcm' par le chemin réel de votre fichier DICOM
dicom_file = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/UNET-TMJ/raw_data/M007/972304'
centre_3d, plus_a_droite_au_centre_3d, plus_en_haut_au_centre_3d = get_3d_coordinates(dicom_file)
print(f"Centre 3D: {centre_3d}")
print(f"Point le plus à droite au centre 3D: {plus_a_droite_au_centre_3d}")
print(f"Point le plus en haut au centre 3D: {plus_en_haut_au_centre_3d}")
# Remplacer 'chemin_vers_votre_fichier.dcm' par le chemin réel de votre fichier DICOM
# dicom_file = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/UNET-TMJ/raw_data/M007/972304'
# centre, plus_a_droite_au_centre, plus_en_haut_au_centre = get_coordinates(dicom_file)

# print(f"Centre: {centre}")
# print(f"Point le plus à droite au centre: {plus_a_droite_au_centre}")
# print(f"Point le plus en haut au centre: {plus_en_haut_au_centre}")


target = [[0,1,0],[0,0,0],[0,0,1]]
source = [plus_a_droite_au_centre_3d,centre_3d,plus_en_haut_au_centre_3d]












ds = pydicom.dcmread(dicom_file)

# Extraire les informations de position et d'orientation
image_position_patient = ds.get('ImagePositionPatient', None)
image_orientation_patient = ds.get('ImageOrientationPatient', None)
pixel_spacing = ds.get('PixelSpacing', None)
slice_thickness = ds.get('SliceThickness', None)
slice_location = ds.get('SliceLocation', None)

print(f"Image Position (Patient): {image_position_patient}")
print(f"Image Orientation (Patient): {image_orientation_patient}")
print(f"Pixel Spacing: {pixel_spacing}")
print(f"Slice Thickness: {slice_thickness}")
print(f"Slice Location: {slice_location}")


source = [[-x for x in image_orientation_patient[:3]],image_orientation_patient[3:]]
print("SOURCE : ",source)
matrix = orientation(target,source,image_position_patient)
print("matrix : ",matrix)




def apply_transformation_to_dicom(dicom_path, output_path, transformation_matrix):
    # Charger le fichier DICOM
    ds = pydicom.dcmread(dicom_path)

    # Extraction des valeurs originales de position et orientation du patient
    original_position = np.array(ds.ImagePositionPatient, dtype=np.float64)
    original_orientation = np.array(ds.ImageOrientationPatient, dtype=np.float64).reshape(2, 3)

    # Appliquer la transformation sur la position du patient
    # La matrice de transformation est de forme 4x4, donc on ajoute un 1 pour la conversion homogène
    transformed_position = np.dot(transformation_matrix, np.append(original_position, 1))[:3]

    # Appliquer la transformation sur l'orientation du patient (pour les deux vecteurs d'orientation)
    transformed_orientation = np.dot(transformation_matrix[:3, :3], original_orientation.reshape(-1, 3).T).T.flatten()

    # Mise à jour des métadonnées du DICOM
    ds.ImagePositionPatient = transformed_position.tolist()
    ds.ImageOrientationPatient = transformed_orientation.tolist()

    # Sauvegarder le fichier DICOM modifié
    ds.save_as(output_path)



def lister_fichiers(dossier,output):
    input = []
    out = []
    for racine, dossiers, fichiers in os.walk(dossier):
        for fichier in fichiers:
            lien = os.path.join(racine, fichier)
            input.append(lien)
            lien = os.path.join(output, fichier)
            out.append(lien)
    return input,out

input_folder = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/UNET-TMJ/raw_data/M007'
output_folder = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/UNET-TMJ/output_rotation'

dossier_input, dossier_out = lister_fichiers(input_folder, output_folder)
print("dossier_input : ",dossier_input)
for i in range(len(dossier_input)):
    input_file = dossier_input[i]
    output_file = dossier_out[i]
    apply_transformation_to_dicom(input_file, output_file, matrix)