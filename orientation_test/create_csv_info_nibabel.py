import nibabel as nib
import os
import pandas as pd
import argparse

def get_nifti_info(file_path):
    # Read the NIfTI file using NiBabel
    image = nib.load(file_path)

    # Get the affine matrix
    affine_matrix = image.affine

    # Get other header information
    header = image.header
    info = {
        "FileName": os.path.basename(file_path),
        "Dimensions": header.get_data_shape(),
        "Size": header.get_data_shape(),
        "Spacing": header.get_zooms(),
        "Origin": affine_matrix[:3, 3],  # Extract the origin from the affine
        "Pixel Type": header.get_data_dtype(),
        "Number of Components per Pixel": header.get_data_dtype().itemsize,  # Approximation
        "Direction": affine_matrix[:3, :3],  # Extract the rotation part of the affine
        "Affine Matrix": affine_matrix
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
        info = get_nifti_info(file)
        nifti_info.append(info)

    # Create a single DataFrame with all the information
    df = pd.DataFrame(nifti_info)
    outpath = os.path.join(args.output, args.name)
    df.to_csv(outpath, index=False)

    return nifti_info

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--input', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Orientation_test/z06_center/image/', help='Input folder')
    parser.add_argument('--output', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Orientation_test/z06_center/', help='Output directory for the aggregated CSV file')
    parser.add_argument('--name', type=str, default='image_info_nibabel.csv', help='name of the CSV')
    args = parser.parse_args()

    # if output folder does not exist, create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    get_nifti_info_folder(args)

