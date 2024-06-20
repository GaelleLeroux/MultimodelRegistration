import os
import argparse
import SimpleITK as sitk
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

def denoise_image(input_file, output_file):
    # Load the NIfTI file
    img = sitk.ReadImage(input_file)
    img_data = sitk.GetArrayFromImage(img)

    # Estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(img_data))

    # Apply Non-Local Means (NLM) denoising
    denoised_img_data = denoise_nl_means(img_data, h=1.15 * sigma_est, fast_mode=True, patch_size=9, patch_distance=7)

    # Convert the denoised numpy array back to a SimpleITK image
    denoised_img = sitk.GetImageFromArray(denoised_img_data)
    denoised_img.CopyInformation(img)

    # Save the denoised image as a new NIfTI file
    output_file = output_file.replace('.nii.gz', 'denoise.nii.gz')
    sitk.WriteImage(denoised_img, output_file)

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)
            print(f'Denoising {input_file}...')
            denoise_image(input_file, output_file)
            print(f'Saved denoised image to {output_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Denoise NIfTI files in a folder.')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing NIfTI files.',default='/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/b4_folder_CBCT_norm/test_percentile=[10,95]_norm=[0,2]')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder to save denoised NIfTI files.',default='/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/b6_folder_CBCT_norm_denoise/test')
    
    args = parser.parse_args()
    
    process_folder(args.input_folder, args.output_folder)
