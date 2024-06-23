import os
import shutil
import argparse

def move_files(input_folder, output_folder, word):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if word in file:
                source_path = os.path.join(root, file)
                destination_path = os.path.join(output_folder, file)
                shutil.move(source_path, destination_path)
                print(f"Moved: {source_path} -> {destination_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move files containing a specific word in their name to another folder.")
    parser.add_argument("--input_folder", type=str, help="The input folder to search for files.",default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/14_Common_Crop (2)/14_Common_Crop")
    parser.add_argument("--output_folder", type=str, help="The output folder to move the files to.",default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a5_folder_14_more_data/d0_CBCT_seg")
    parser.add_argument("--word", type=str, help="The word to search for in the file names.",default="CBCT_Seg")

    args = parser.parse_args()

    move_files(args.input_folder, args.output_folder, args.word)
