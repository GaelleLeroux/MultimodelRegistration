import argparse
import os
import SimpleITK as sitk

def normalize_image(image):
    """Normalize the image data to range [0, 1]."""
    image_array = sitk.GetArrayFromImage(image)
    min_val = image_array.min()
    max_val = image_array.max()
    print("min : ",min_val)
    print("max : ",max_val)
    normalized_array = (image_array - min_val) / (max_val - min_val) if max_val > min_val else image_array
    return sitk.GetImageFromArray(normalized_array)

def process_images(input_folder, output_folder, suffix):
    """Process all .nii.gz images in the input folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_folder, filename)
            img = sitk.ReadImage(input_path)
            
            # Normalize the image data
            normalized_img = normalize_image(img)
            
            # Copy original metadata to the normalized image
            print("filename : ",filename)
            normalized_img.CopyInformation(img)
            
            # Save the normalized image with the new suffix
            output_filename = filename.replace('.nii.gz', f'_{suffix}.nii.gz')
            output_path = os.path.join(output_folder, output_filename)
            sitk.WriteImage(normalized_img, output_path)
            print(f'Saved normalized image to {output_path}')

def main():
    parser = argparse.ArgumentParser(description='Normalize NIfTI images and save with a new suffix.')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing .nii.gz images.', default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/a2_folder_MRI/test")
    parser.add_argument('--output_folder', type=str, help='Path to the output folder to save normalized images.', default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/a4_folder_MRI_norm/test")
    parser.add_argument('--suffix', type=str, help='Suffix to add to the output filenames.',default="norm")
    
    args = parser.parse_args()
    
    process_images(args.input_folder, args.output_folder, args.suffix)

if __name__ == '__main__':
    main()
