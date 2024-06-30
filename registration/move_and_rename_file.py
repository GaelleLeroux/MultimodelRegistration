import os
import shutil
import argparse

def rename_and_move_files(input_folder, output_folder, old_suffix, new_suffix):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if old_suffix in file:
                new_file = file.replace(old_suffix, new_suffix)
                source_path = os.path.join(root, file)
                destination_path = os.path.join(output_folder, new_file)
                shutil.copy2(source_path, destination_path)
                print(f"Copied: {source_path} -> {destination_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename and move files from input folder to output folder.")
    parser.add_argument("--input_folder", type=str, help="The input folder to search for files.", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/14_Common_Crop (2)/14_Common_Crop")
    parser.add_argument("--output_folder", type=str, help="The output folder to move the files to.", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a5_folder_14_more_data/d0_CBCT_seg")
    parser.add_argument("--old_suffix", type=str, help="The old suffix to be replaced in the file names.", default="_MR_MA_apply_MA.nii.gz")
    parser.add_argument("--new_suffix", type=str, help="The new suffix to replace the old one.", default="_MR_OC.nii.gz")

    args = parser.parse_args()

    rename_and_move_files(args.input_folder, args.output_folder, args.old_suffix, args.new_suffix)
