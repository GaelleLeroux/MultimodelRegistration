import argparse
import SimpleITK as sitk
import sys, os, time
import numpy as np

import nibabel as nib
fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)
import os
import glob

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



def ResampleImage(image,label, transform):
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
    
    
    resample_label = sitk.ResampleImageFilter()
    resample_label.SetReferenceImage(image)
    resample_label.SetTransform(transform)
    resample_label.SetInterpolator(sitk.sitkLinear)
    resample_label.SetSize(new_size)
    resample_label.SetDefaultPixelValue(0)

    # Set New Origin
    orig_origin = np.array(image.GetOrigin())
    # apply transform to the origin
    orig_center = np.array(
        image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize()) / 2.0)
    )
    # new_center = np.array(target.TransformContinuousIndexToPhysicalPoint(np.array(target.GetSize())/2.0))
    new_origin = orig_origin - orig_center
    resample.SetOutputOrigin(new_origin)
    resample_label.SetOutputOrigin(new_origin)

    return resample.Execute(image), resample_label.Execute(label)


def center(args):
    # train_images = sorted(glob.glob(args.input_image))
    # train_labels = sorted(glob.glob(args.input_label))
    
    train_images = sorted(glob.glob(os.path.join(args.input_image, "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.input_label, "*.nii.gz")))

    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    print("data_dicts : ",data_dicts)

    input_folder_image = args.input_image
    output_folder_image = args.output_image
    
    input_folder_label = args.input_label
    output_folder_label = args.output_label
    
    # path = lister_fichiers(input_folder)
    # path = ["/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/imagesTr/M004_MRI.nii.gz"]
    # output_folder = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/test_orientation/"
    
    for item in data_dicts :

        input_image = item['image']
        input_label = item['label']
        
        # print(f"input_image : {input_image}")
        # print(f"input_label : {input_label}")

        img = sitk.ReadImage(input_image)
        label = sitk.ReadImage(input_label)

        # Translation to center volume
        T = -np.array(
            img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0)
        )
        translation = sitk.TranslationTransform(3)
        translation.SetOffset(T.tolist())
        # print("translation.GetInverse() : ",translation.GetInverse())
        # print("T: ",T)
        # print("T.tolist() : ",T.tolist())

        img_trans,label_trans = ResampleImage(img, label,translation.GetInverse())
        img_out = img_trans
        label_out = label_trans

        # Write Scan
        dir_scan = os.path.dirname(input_image.replace(input_image, output_folder_image))
        if not os.path.exists(dir_scan):
            os.makedirs(dir_scan)
        
        file_outpath = os.path.join(dir_scan, os.path.basename(input_image))
        
        if not os.path.exists(file_outpath):
            sitk.WriteImage(img_out, file_outpath)
            
        #save label
            
        dir_scan = os.path.dirname(input_label.replace(input_label, output_folder_label))
        if not os.path.exists(dir_scan):
            os.makedirs(dir_scan)
        
        file_outpath = os.path.join(dir_scan, os.path.basename(input_label))
        
        if not os.path.exists(file_outpath):
            sitk.WriteImage(label_out, file_outpath)
            
        print(f"fichier : {os.path.basename(input_image)} finit d'etre traite")
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)
        print(f"""<filter-progress>{2}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)
        print(f"""<filter-progress>{0}</filter-progress>""")
        sys.stdout.flush()
        time.sleep(0.2)




if __name__ == "__main__":
    
    print("PRE ASO")

    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--input_image', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/imagesTr/', help='Input folder')
    parser.add_argument('--output_image', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/z06_center/image/', help='Output directory for the aggregated CSV file')
    parser.add_argument('--input_label', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/labelsTr/', help='Input folder')
    parser.add_argument('--output_label', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/z06_center/label/', help='Output directory for the aggregated CSV file')
    args = parser.parse_args()
    
    print("args  : ",args)
   
    # img = sitk.ReadImage("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/imagesTr/M004_MRI.nii.gz")
    # target_vector = (1, 0, 0)  # Dans cet exemple, c'est déjà aligné avec l'axe des x
    # img_aligned = align_image_to_vector(img, target_vector)
    center(args)
