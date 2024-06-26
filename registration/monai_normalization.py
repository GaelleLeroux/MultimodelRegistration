import os
import argparse
from monai.transforms import ScaleIntensityRangePercentiles, LoadImage, SaveImage

def process_images(input_folder, output_folder, suffix, lower, upper, b_min, b_max, clip):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize MONAI's SaveImage
    saver = SaveImage(output_dir=output_folder, output_postfix=suffix, output_ext='.nii.gz', resample=False, separate_folder=False)

    # Iterate through all .nii.gz files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            # Load the image using MONAI
            input_path = os.path.join(input_folder, filename)
            img, meta_data = LoadImage(image_only=False)(input_path)
            
            # Normalize the image
            transform = ScaleIntensityRangePercentiles(lower, upper, b_min, b_max, clip)
            img_normalized = transform(img)

            # Save the new image using MONAI's SaveImage
            meta_data["filename_or_obj"] = filename  # Ensure the filename is passed to the saver
            saver(img_normalized, meta_data)
            
            print(f"Processed and saved: {os.path.join(output_folder, filename.replace('.nii.gz', f'{suffix}.nii.gz'))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize .nii.gz images using ScaleIntensityRangePercentiles from MONAI")
    parser.add_argument("--input_folder", type=str, help="Folder containing the input .nii.gz files", default="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Registration/B11_B12/a01_MRI_inv/")
    parser.add_argument("--output_folder", type=str, help="Folder to save the processed .nii.gz files", default="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Registration/B11_B12/a2_MRI_inv_norm/test_percentile=[10,95]_norm=[0,100]_monai/")
    parser.add_argument("--suffix", type=str, help="Suffix to add to the processed file names", default="Mnorm")
    parser.add_argument("--lower", type=int, help="Percentile lower bound for intensity scaling", default=10)
    parser.add_argument("--upper", type=int, help="Percentile upper bound for intensity scaling", default=95)
    parser.add_argument("--b_min", type=int, help="Output intensity minimum value", default=0)
    parser.add_argument("--b_max", type=int, help="Output intensity maximum value", default=100)
    parser.add_argument("--clip", type=bool, default=False, help="Whether to clip values outside the range")

    args = parser.parse_args()

    output_path = os.path.join(args.output_folder,f"test_percentile=[{args.lower},{args.upper}]_norm=[{args.b_min},{args.b_max}]")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    process_images(args.input_folder, output_path, args.suffix, args.lower, args.upper, args.b_min, args.b_max, args.clip)
