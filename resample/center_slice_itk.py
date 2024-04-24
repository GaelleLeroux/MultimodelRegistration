import SimpleITK as sitk
import numpy as np
import argparse
import os

def center_image_fully_sitk(img_filename,output):
    # Convertir SimpleITK.Image en numpy array
    img = sitk.ReadImage(img_filename)
    data = sitk.GetArrayFromImage(img)

    # Obtenir les dimensions
    sz, sy, sx = data.shape

    # Identifier les tranches non-noires dans les trois dimensions
    non_black_x = np.any(data, axis=(0, 1))
    non_black_y = np.any(data, axis=(0, 2))
    non_black_z = np.any(data, axis=(1, 2))

    # Calculer les indices minimaux et maximaux non-noirs pour chaque axe
    xmin, xmax = np.where(non_black_x)[0][[0, -1]] if np.any(non_black_x) else (0, sx)
    ymin, ymax = np.where(non_black_y)[0][[0, -1]] if np.any(non_black_y) else (0, sy)
    zmin, zmax = np.where(non_black_z)[0][[0, -1]] if np.any(non_black_z) else (0, sz)

    # Déterminer les centres des indices non-noirs
    center_x, center_y, center_z = (xmin + xmax) // 2, (ymin + ymax) // 2, (zmin + zmax) // 2

    # Calculer les décalages nécessaires pour centrer l'image
    shift_x, shift_y, shift_z = sx//2 - center_x, sy//2 - center_y, sz//2 - center_z

    # Appliquer les décalages à chaque dimension
    data = np.roll(data, shift_x, axis=2)
    data = np.roll(data, shift_y, axis=1)
    data = np.roll(data, shift_z, axis=0)

    # Convertir le numpy array centré en SimpleITK.Image
    centered_img = sitk.GetImageFromArray(data)
    centered_img.CopyInformation(img)  # Copier les informations de l'image originale

    writer = sitk.ImageFileWriter()
    writer.SetFileName(output)
    writer.UseCompressionOn()
    writer.Execute(centered_img)

def type_file(filename):
    type_file = '.nii'
    base = os.path.splitext(filename)[0]
    # If the file has a double extension (commonly .nii.gz), remove the second extension
    if base.endswith('.nii'):
        base = os.path.splitext(base)[0]
        type_file='.nii.gz'
    
    return type_file
        
    

def main(args):
    input_folder = args.input
    # Get all nifti files in the folder
    nifti_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nifti_files.append(os.path.join(root, file))
                
    for file in nifti_files :
        outpath = file.replace(input_folder,args.output)
        extension_scan = type_file(file)
        output = outpath.split(extension_scan)[0]+"_"+args.suffix+extension_scan
        center_image_fully_sitk(file,output)


if __name__=="__main__":
    # SIZE AND SPACING TO RESAMPLE ARE HARD WRITTEN IN THE LINE 24
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--input',default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/RESAMPLE/a02_resample_non_centre/',type=str, help='Input folder')
    parser.add_argument('--output',default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/RESAMPLE/a03_recenter_2/', type=str, help='Output folder')
    parser.add_argument('--suffix', default="center",type=str, help='Input folder')
    # /home/luciacev/Documents/Gaelle/MultimodelRegistration/resample/resample.py
    args = parser.parse_args()


    main(args)
    # center_image_fully_sitk('/home/luciacev/Documents/Gaelle/Data/MultimodelReg_2/RESAMPLE/a02_resample_non_centre/M004_OR.nii')