import os
import SimpleITK as sitk
import argparse

def calculate_new_origin(image):
    """
    Calculate the new origin to center the image in the Slicer viewport.
    """
    size = image.GetSize()
    spacing = image.GetSpacing()
    new_origin = [-(size[d] * spacing[d]) / 2 for d in range(len(size))]
    return tuple(new_origin)

def modify_image_properties(nifti_file_path, new_direction, new_origin, output_file_path=None):
    """
    Read a NIfTI file, change its Direction and Origin, and optionally save the modified image.
    """
    image = sitk.ReadImage(nifti_file_path)
    print("Original Direction:", image.GetDirection())
    print("Original Origin:", image.GetOrigin())
    
    image.SetDirection(new_direction)
    image.SetOrigin(new_origin)
    
    print("New Direction:", image.GetDirection())
    print("New Origin:", image.GetOrigin())
    
    if output_file_path:
        sitk.WriteImage(image, output_file_path)
        print(f"Modified image saved to {output_file_path}")

    return image

def main(args):
    new_direction = tuple(map(float, args.direction.split(',')))  # Assumes direction as comma-separated values
    new_origin = tuple(map(float, args.origin.split(',')))  # Assumes origin as comma-separated values
    input_folder = args.input
    output_folder = args.output if args.output else input_folder  # Default to input folder if no output folder is provided.

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all nifti files in the folder
    nifti_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nifti_files.append(os.path.join(root, file))

    # Process each file
    for file_path in nifti_files:
        filename = os.path.basename(file_path)
        output_file_path = os.path.join(output_folder, f"modified_{filename}")
        modify_image_properties(file_path, new_direction, new_origin, output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify NIfTI file directions.")
    parser.add_argument('--input', default = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/More_DATA_to_center/MRs_Anonymized_left/', help='Path to the input folder containing NIfTI files.')
    parser.add_argument('--direction', default = "0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0" ,help='New direction for the NIfTI files, specified as a comma-separated string of floats.')
    parser.add_argument('--origin', default = "0.0, 0.0, 0.0", help='New origin for the NIfTI files, specified as a comma-separated string of floats.')
    parser.add_argument('--output', default = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/More_DATA_to_center/left_oriented/', help='Path to the output folder where modified NIfTI files will be saved.')
    args = parser.parse_args()
    main(args)
