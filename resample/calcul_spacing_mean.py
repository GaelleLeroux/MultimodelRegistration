import SimpleITK as sitk
import os
import pandas as pd
import argparse
import numpy as np

def get_info(file_path):
    # Read the NIfTI file
    image = sitk.ReadImage(file_path)

    # Get information
    
    spacing = image.GetSpacing()
        

    return spacing[0],spacing[1],spacing[2]

def calcul_mean(args):
    input_folder = args.input
    # Get all nifti files in the folder
    nifti_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nifti_files.append(os.path.join(root, file))

    # Get nifti info for every nifti file
    x_list = []
    y_list = []
    z_list = []
    nifti_info = []
    for file in nifti_files:
        x_tempo, y_tempo, z_tempo = get_info(file)
        x_list.append(x_tempo)
        y_list.append(y_tempo)
        z_list.append(z_tempo)
    

    print("x_list : ",x_list)
    print("y_list : ",y_list)
    print("z_list : ",z_list)

    print("*"*100)
    print("Spacing moyen de : ",np.mean(x_list)," , ",np.mean(y_list)," , ",np.mean(z_list))
    return (np.mean(x_list),np.mean(y_list),np.mean(z_list))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--input', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/imagesTr/', help='Input folder')
    args = parser.parse_args()


    calcul_mean(args)
