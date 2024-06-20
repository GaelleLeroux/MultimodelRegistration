import argparse
import os
import SimpleITK as sitk

def enhance_contrast(image):
    """Enhance the contrast of the image while keeping normalization between 0 and 1."""
    # Use SimpleITK's Adaptive Histogram Equalization to enhance contrast
    enhanced_image = sitk.AdaptiveHistogramEqualization(image, alpha=0.1, beta=0.01)
    
    # Ensure the output image is still normalized between 0 and 1
    array = sitk.GetArrayFromImage(enhanced_image)
    min_val = array.min()
    max_val = array.max()
    if max_val > min_val:
        normalized_array = (array - min_val) / (max_val - min_val)
    else:
        normalized_array = array
    return sitk.GetImageFromArray(normalized_array)

def process_images(input_folder, output_folder, suffix):
    """Process all .nii.gz images in the input folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_folder, filename)
            img = sitk.ReadImage(input_path)
            
            # Enhance the contrast of the image
            enhanced_img = enhance_contrast(img)
            
            # Copy original metadata to the enhanced image
            enhanced_img.CopyInformation(img)
            
            # Save the enhanced image with the new suffix
            output_filename = filename.replace('.nii.gz', f'_{suffix}.nii.gz')
            output_path = os.path.join(output_folder, output_filename)
            sitk.WriteImage(enhanced_img, output_path)
            print(f'Saved enhanced image to {output_path}')

def main():
    parser = argparse.ArgumentParser(description='Enhance contrast of NIfTI images and save with a new suffix.')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing .nii.gz images.', default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/a4_folder_MRI_invert_norm/test")
    parser.add_argument('--output_folder', type=str, help='Path to the output folder to save normalized images.', default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/a5_folder_MRI_invert_norm_contrast/test")
    parser.add_argument('--suffix', type=str, help='Suffix to add to the output filenames.',default="contrast")
    
    args = parser.parse_args()
    
    process_images(args.input_folder, args.output_folder, args.suffix)

if __name__ == '__main__':
    main()
