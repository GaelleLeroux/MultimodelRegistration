import nibabel as nib
import numpy as np

# Exemple d'utilisation :
# center_image_slice_by_slice('/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/RESAMPLE/a02_resample/M004_OR.nii')


def center_image_fully(nii_file):
    # Charger l'image NIfTI
    img = nib.load(nii_file)
    data = img.get_fdata()
    
    # Préparer le nouveau tableau de données avec la même forme
    centered_data = np.zeros(data.shape)
    
    # Obtenir les dimensions
    sx, sy, sz = data.shape

    # Identifier les tranches non-noires dans les trois dimensions
    non_black_x = np.any(data, axis=(1, 2))
    non_black_y = np.any(data, axis=(0, 2))
    non_black_z = np.any(data, axis=(0, 1))

    # Calculer les indices minimaux et maximaux non-noirs pour chaque axe
    xmin, xmax = np.where(non_black_x)[0][[0, -1]] if np.any(non_black_x) else (0, sx)
    ymin, ymax = np.where(non_black_y)[0][[0, -1]] if np.any(non_black_y) else (0, sy)
    zmin, zmax = np.where(non_black_z)[0][[0, -1]] if np.any(non_black_z) else (0, sz)

    # Déterminer les centres des indices non-noirs
    center_x, center_y, center_z = (xmin + xmax) // 2, (ymin + ymax) // 2, (zmin + zmax) // 2

    # Calculer les décalages nécessaires pour centrer l'image
    shift_x, shift_y, shift_z = sx//2 - center_x, sy//2 - center_y, sz//2 - center_z

    # Appliquer les décalages à chaque dimension
    data = np.roll(data, shift_x, axis=0)
    data = np.roll(data, shift_y, axis=1)
    data = np.roll(data, shift_z, axis=2)
    
    # Créer une nouvelle image NIfTI avec les données centrées
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    
    # Sauvegarder l'image centrée
    nib.save(new_img, '/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/RESAMPLE/a03_recenter/M004_OR_Center_fully.nii')
    
center_image_fully('/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/RESAMPLE/a02_resample/M004_OR.nii')