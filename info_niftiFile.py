import SimpleITK as sitk
import os
import pandas as pd
import argparse

def get_nifti_info(file_path):
    # Read the NIfTI file
    image = sitk.ReadImage(file_path)
    
    statistics_filter = sitk.StatisticsImageFilter()
    statistics_filter.Execute(image)
    min_pixel_value = statistics_filter.GetMinimum()
    max_pixel_value = statistics_filter.GetMaximum()

    # Get information
    info = {
        "FileName": os.path.basename(file_path),  # Ajoutez le nom du fichier ici
        "Dimensions": image.GetDimension(),
        "Size": image.GetSize(),
        "Spacing": image.GetSpacing(),
        "Origin": image.GetOrigin(),
        "Direction": image.GetDirection(),
        "Pixel Type": sitk.GetPixelIDValueAsString(image.GetPixelID()),
        "Min Pixel Value": min_pixel_value,  
        "Max Pixel Value": max_pixel_value,
        "Number of Components per Pixel": image.GetNumberOfComponentsPerPixel()
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

    # Créez un seul DataFrame avec toutes les informations
    df = pd.DataFrame(nifti_info)
    outpath = os.path.join(args.output, args.csv)
    df.to_csv(outpath, index=False)

    return nifti_info

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--input', type=str, default='Data_test_AREG', help='Input folder')
    parser.add_argument('--output', type=str, default='.', help='Output directory for the aggregated CSV file')
    parser.add_argument('--csv', type=str, default='Nifti_info.csv', help='Output directory for the aggregated CSV file')
    args = parser.parse_args()

    # if output folder does not exist, create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    get_nifti_info_folder(args)
