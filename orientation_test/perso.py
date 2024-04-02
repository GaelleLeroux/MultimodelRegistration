import nibabel as nib
import numpy as np

nifti_img = nib.load("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/imagesTr/M004_MRI.nii.gz")

# t1_hdr = t1_img.nifti_file_path
# print(t1_hdr)
x_scale = np.linalg.norm(nifti_img.affine[:,0])
y_scale = np.linalg.norm(nifti_img.affine[:,1])
z_scale = np.linalg.norm(nifti_img.affine[:,2])
print(x_scale, y_scale, z_scale)

spacing = [x_scale,y_scale,z_scale]
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


scaling_matrix = np.diag(spacing)
new_rotation_scaled = scaling_matrix @ new_rotation
# Reconstruire la matrice affine avec la nouvelle rotation
new_affine = np.eye(4)
new_affine[:3, :3] = new_rotation
new_affine[:3, 3] = translation


new_nifti_img = nib.Nifti1Image(nifti_img.get_fdata(), new_affine)

# Chemin où sauvegarder le nouveau fichier NIfTI
new_nifti_file_path = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/test_orientation/M004_orientation4.nii.gz"
# Sauvegarder le nouveau fichier NIfTI
nib.save(new_nifti_img, new_nifti_file_path)

print("Nouvelle matrice affine pour être parallèle au plan z:\n", new_affine)