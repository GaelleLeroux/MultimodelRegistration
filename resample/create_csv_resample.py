import SimpleITK as sitk
import os
import pandas as pd
import argparse

def get_nifti_info(file_path,output_resample):
    # Read the NIfTI file
    image = sitk.ReadImage(file_path)

    # Get information
    info = {
        "in": file_path, 
        "out" : file_path.replace(os.path.dirname(file_path),output_resample),
        "size": image.GetSize(),
        "Spacing": image.GetSpacing(),
    }

    return info

def get_nifti_info_folder(args):
    input_folder = args.input
    # Get all nifti files in the folder
    nifti_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nifti_files.append(os.path.join(root, file))

    # Get nifti info for every nifti file
    nifti_info = []
    for file in nifti_files:
        info = get_nifti_info(file,args.output_resample)
        nifti_info.append(info)

    # Créez un seul DataFrame avec toutes les informations
    df = pd.DataFrame(nifti_info)
    outpath = os.path.join(args.output, args.name)
    df.to_csv(outpath, index=False)

    return nifti_info

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--input', type=str, default='/home/lucia/Documents/Gaelle/Data/Developpement_extension_MRI2CBCT/test', help='Input folder')
    parser.add_argument('--output', type=str, default='/home/lucia/Documents/Gaelle/Data/Developpement_extension_MRI2CBCT/test2', help='Output directory for the aggregated CSV file')
    parser.add_argument('--name', type=str, default='resample_cbct.csv', help='name of the CSV')
    parser.add_argument('--output_resample', type=str, default='/home/lucia/Documents/Gaelle/Data/Developpement_extension_MRI2CBCT/test2', help='Output directory were the file will be save after resample')
    args = parser.parse_args()

    # if output folder does not exist, create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    if not os.path.exists(args.output_resample):
        os.makedirs(args.output_resample)

    get_nifti_info_folder(args)
