import argparse
import os
import numpy as np
import SimpleITK as sitk

def compute_center_of_mass(image):
    """
    Compute the center of mass of the non-zero voxels in the given image.
    """
    arr = sitk.GetArrayFromImage(image)
    non_zero_coords = np.argwhere(arr > 0)
    center_of_mass = non_zero_coords.mean(axis=0)
    # Convert from numpy (z, y, x) to SimpleITK (x, y, z)
    center_of_mass = np.flip(center_of_mass)
    print("center_of_mass : ",center_of_mass)
    return center_of_mass

def compute_center_of_mass_itk(image):
    """
    Compute the center of mass of the non-zero voxels in the given image using SimpleITK.
    """
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(image)
    # Assuming label 1 is the segmentation of interest
    center_of_mass = label_shape_filter.GetCentroid(1)
    print("center_of_mass : ", center_of_mass)
    return center_of_mass

def create_transform_matrix(center_of_mass):
    """
    Create a transformation matrix that translates (0, 0, 0) to the center of mass.
    """
    transform = sitk.AffineTransform(3)
    inverse_translation = [-coord for coord in center_of_mass]
    transform.SetTranslation(inverse_translation)
    return transform

def save_transform(transform, filename):
    """
    Save the given transform to a .tfm file.
    """
    sitk.WriteTransform(transform, filename)

def process_files(input_folder, output_folder):
    """
    Process all .nii.gz files in the input folder, compute their center of mass,
    create transformation matrices, and save them to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    idx = 0
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            idx+=1
            filepath = os.path.join(input_folder, filename)
            image = sitk.ReadImage(filepath)
            center_of_mass = compute_center_of_mass_itk(image)
            transform = create_transform_matrix(center_of_mass)
            base = os.path.splitext(filename)[0].split("_CBCT_")[0]
            if base.endswith('.nii'):
                base = os.path.splitext(base)[0]
            output_filepath = os.path.join(output_folder, base + '_MA.tfm')
            save_transform(transform, output_filepath)
            print(f"Processed {filename} and saved transform to {output_filepath}")
            if idx>10:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process .nii.gz files to compute center of mass and save transformation matrices.')
    parser.add_argument('--input_folder', type=str, help='The folder containing the input .nii.gz files. Here the segmentation (label2) of the CBCT')
    parser.add_argument('--output_folder', type=str, help='The folder to save the output .tfm files.')

    args = parser.parse_args()

    process_files(args.input_folder, args.output_folder)
