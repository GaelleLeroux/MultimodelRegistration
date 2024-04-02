#!/usr/bin/env python-real

import argparse
import SimpleITK as sitk
import sys, os, time
import numpy as np

import nibabel as nib
fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)
import os

def lister_fichiers(dossier):
    # Liste pour stocker les chemins des fichiers
    fichiers = []
    
    # Parcourir le dossier et ses sous-dossiers
    for racine, dossiers, fichiers_dans_dossier in os.walk(dossier):
        for nom_fichier in fichiers_dans_dossier:
            # Construire le chemin complet du fichier
            chemin_complet = os.path.join(racine, nom_fichier)
            # Ajouter le chemin du fichier à la liste
            fichiers.append(chemin_complet)
    
    return fichiers



def ResampleImage(image, transform):
    """
    Resample image using SimpleITK

    Parameters
    ----------
    image : SimpleITK.Image
        Image to be resampled
    target : SimpleITK.Image
        Target image
    transform : SimpleITK transform
        Transform to be applied to the image.

    Returns
    -------
    SimpleITK image
        Resampled image.
    """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(image)
    resample.SetTransform(transform)
    resample.SetInterpolator(sitk.sitkLinear)
    orig_size = np.array(image.GetSize(), dtype=int)
    ratio = 1
    new_size = orig_size * ratio
    new_size = np.ceil(new_size).astype(int)  #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetDefaultPixelValue(0)

    # Set New Origin
    orig_origin = np.array(image.GetOrigin())
    # apply transform to the origin
    orig_center = np.array(
        image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) / 2.0)
    )
    # new_center = np.array(target.TransformContinuousIndexToPhysicalPoint(np.array(target.GetSize())/2.0))
    new_origin = orig_origin - orig_center
    resample.SetOutputOrigin(new_origin)

    return resample.Execute(image)


def center():


    input_folder = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/imagesTr/"
    output_folder = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/test_orientation/"
    
    path = lister_fichiers(input_folder)
    path = ["/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/imagesTr/M004_MRI.nii.gz"]
    output_folder = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/test_orientation/"
    
    for input_file in path :

        # input_file = input_files

        img = sitk.ReadImage(input_file)

        # Translation to center volume
        T = -np.array(
            img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0)
        )
        translation = sitk.TranslationTransform(3)
        translation.SetOffset(T.tolist())
        print("translation.GetInverse() : ",translation.GetInverse())
        print("T: ",T)
        print("T.tolist() : ",T.tolist())

        img_trans = ResampleImage(img, translation.GetInverse())
        img_out = img_trans

        # Write Scan
        dir_scan = os.path.dirname(input_file.replace(input_folder, output_folder))
        if not os.path.exists(dir_scan):
            os.makedirs(dir_scan)

        file_outpath = os.path.join(dir_scan, os.path.basename(input_file))
        # file_outpath = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/test_orientation/M004_orientation.nii.gz"
        if not os.path.exists(file_outpath):
            sitk.WriteImage(img_out, file_outpath)
            
        print(f"fichier : {os.path.basename(input_file)} finit d'etre traite")
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)
        print(f"""<filter-progress>{2}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)
        
        
def rotation():
    
    
    
    input_folder = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/center/"
    output_folder = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/test_orientation/"
    input_file = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/center/M004_MRI.nii.gz"
    
    nifti_img = nib.load(input_file)
    affine = nifti_img.affine

    # Extraire la rotation et la translation de la matrice affine
    rotation = affine[:3, :3]
    translation = affine[:3, 3]

    # Décomposer la rotation actuelle
    u, _, vh = np.linalg.svd(rotation, full_matrices=True)

    # Calculer la nouvelle matrice de rotation pour aligner avec l'axe z
    # Dans ce cas, nous alignons simplement l'axe z sans changer x et y
    # Donc, nous pouvons utiliser u et vh directement
    new_rotation = np.dot(u, vh)

    # Assurer que la matrice de rotation est bien orientée (déterminant = 1)
    if np.linalg.det(new_rotation) < 0:
        new_rotation[:, 0] = -new_rotation[:, 0]


    # scaling_matrix = np.diag(spacing)
    # new_rotation_scaled = scaling_matrix @ new_rotation
    # Reconstruire la matrice affine avec la nouvelle rotation
    # new_affine = np.eye(4)
    # new_affine[:3, :3] = new_rotation
    # new_affine[:3, 3] = translation

    print("new_rotation : ",new_rotation)
    
    TransformMatrixsitk = sitk.Euler3DTransform()
    # TransformMatrixsitk.SetTranslation(TransformMatrixBis[:3, 3].tolist())
    TransformMatrixsitk.SetMatrix(new_affine.flatten().tolist())
    img = sitk.ReadImage(input_file)
    img_out = ResampleImageRotation(img,TransformMatrixsitk)
   


    # Chemin où sauvegarder le fichier de transformation
    chemin_fichier_tfm = os.path.join(output_folder,"transformation2.tfm")

    # Sauvegarder la transformation dans un fichier .tfm
    sitk.WriteTransform(TransformMatrixsitk, chemin_fichier_tfm)
    
    # Write Scan
    dir_scan = os.path.dirname(input_file.replace(input_folder, output_folder))
    if not os.path.exists(dir_scan):
        os.makedirs(dir_scan)

    file_outpath = os.path.join(dir_scan, os.path.basename(input_file))
    # file_outpath = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/test_orientation/M004_orientation.nii.gz"
    if not os.path.exists(file_outpath):
        sitk.WriteImage(img_out, file_outpath)

    print("ok")
    
def ResampleImageRotation(image, transform):
    """
    Resample image using SimpleITK

    Parameters
    ----------
    image : SimpleITK.Image
        Image to be resampled
    target : SimpleITK.Image
        Target image
    transform : SimpleITK transform
        Transform to be applied to the image.

    Returns
    -------
    SimpleITK image
        Resampled image.
    """
    # Create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkLinear)
    # resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    return resampler.Execute(image)


def align_image_to_vector(img, target_vector):
    # Obtenez l'orientation actuelle de l'image
    current_direction = np.array(img.GetDirection()).reshape(3, 3)
    
    # Déterminez l'orientation cible en alignant l'axe principal avec target_vector
    target_direction = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Exemple simplifié
    
    # Calculez la matrice de rotation nécessaire
    rotation_matrix = np.dot(target_direction, np.linalg.inv(current_direction))
    print("rotation matrix : ",rotation_matrix)
    print("rotation_matrix.flatten() : ",rotation_matrix.flatten())
    
    matrix_4x4 = np.eye(4)

    # Remplacer les 3 premières lignes et colonnes par la matrice de rotation 3x3
    matrix_4x4[:3, :3] = rotation_matrix
    print("matrix_4x4 : ",matrix_4x4)
    
    # Créez la transformation de rotation
    # transform = sitk.AffineTransform(3)
    # transform.SetMatrix(matrix_4x4.flatten())
    
    # resample = sitk.ResampleImageFilter()
    # resample.SetReferenceImage(img)
    # resample.SetTransform(transform)
    # resample.SetInterpolator(sitk.sitkLinear)
    # img_out = resample.Execute(img)
    
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix_4x4[:3, :3].flatten())
    transform.SetTranslation(matrix_4x4[:3, 3])

    # Étape 3 : Appliquer la transformation à l'image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetTransform(transform)
    resampler.SetSize(img.GetSize())
    resampler.SetOutputSpacing(img.GetSpacing())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(0)  # ou toute autre valeur par défaut appropriée
    resampler.SetInterpolator(sitk.sitkLinear)
    img_out = resampler.Execute(img)
    
    file_outpath = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/test_orientation/M004_orientation.nii.gz"
    if not os.path.exists(file_outpath):
        sitk.WriteImage(img_out, file_outpath)
        
        
        
        
        
    # Translation to center volume
    T = -np.array(
        img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0)
    )
    translation = sitk.TranslationTransform(3)
    translation.SetOffset(T.tolist())

    img_trans = ResampleImage(img_out, translation.GetInverse())
    img_cent = img_trans
    file_outpath = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/test_orientation/M004_or_cent.nii.gz"
    if not os.path.exists(file_outpath):
        sitk.WriteImage(img_cent, file_outpath)
        
    # img_trans = ResampleImage(img, translation.GetInverse())
    # img_just_cent = img_trans
    # file_outpath = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/test_orientation/M004_cent.nii.gz"
    # if not os.path.exists(file_outpath):
    #     sitk.WriteImage(img_just_cent, file_outpath)

    
    

if __name__ == "__main__":

    print("PRE ASO")

   
    # img = sitk.ReadImage("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/imagesTr/M004_MRI.nii.gz")
    # target_vector = (1, 0, 0)  # Dans cet exemple, c'est déjà aligné avec l'axe des x
    # img_aligned = align_image_to_vector(img, target_vector)
    rotation()
