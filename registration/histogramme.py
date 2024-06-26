import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def create_histogram(nifti_file, output_folder):
    # Load the NIfTI file
    img = nib.load(nifti_file)
    data = img.get_fdata()
    
    # Count the number of zeros
    num_zeros = np.sum(data == 0)
    
    # Exclude zeros from the data
    data_non_zero = data[data != 0]
    
    # Create histogram
    plt.figure()
    plt.hist(data_non_zero, bins=50, color='blue', alpha=0.7)
    plt.title(f'Histogram\nNumber of zeros: {num_zeros}')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    
    # Save histogram
    filename = os.path.basename(nifti_file).replace('.nii.gz', '.png')
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path)
    plt.close()

def process_folder(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each NIfTI file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            nifti_file = os.path.join(input_folder, filename)
            create_histogram(nifti_file, output_folder)
            print(f'Processed {filename}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate histograms for NIfTI files")
    parser.add_argument("--input_folder", type=str, help="Input folder containing NIfTI files")
    parser.add_argument("--output_folder", type=str, help="Output folder for histograms")

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)
