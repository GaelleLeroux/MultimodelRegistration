import os
import pandas as pd
import argparse
import re

def find_mri_and_seg_files(mri_dir, seg_dir):
    mri_files = []
    seg_files = []

    # Regular expression to match the file pattern with any number in place of X
    mri_pattern = re.compile(r"(_MR_Crop_slice_\d+\.png)$")
    seg_pattern = re.compile(r"(_MR_Seg_clean_slice_\d+\.png)$")

    # Traverse the MRI directory
    for root, dirs, files in os.walk(mri_dir):
        for file in files:
            if mri_pattern.search(file):
                mri_files.append(os.path.join(root, file))

    # Traverse the segmentation directory
    for root, dirs, files in os.walk(seg_dir):
        for file in files:
            if seg_pattern.search(file):
                seg_files.append(os.path.join(root, file))
                

    return mri_files, seg_files

def create_csv_referencing_mri_and_seg(mri_files, seg_files, csv_file):
    data = {
        "mri": [],
        "seg": []
    }

    for mri in mri_files:
        match = re.search(r"(.*)_MR_Crop_slice_(\d+)\.png$", os.path.basename(mri))
        if match:
            patient_name = match.group(1)
            slice_num = match.group(2)
            seg_pattern = f"{patient_name}_MR_Seg_clean_slice_{slice_num}.png"
            seg = next((s for s in seg_files if os.path.basename(s) == seg_pattern), None)
        
            if seg:
                data["mri"].append(mri)
                data["seg"].append(seg)

    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"CSV file created: {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='Create a CSV referencing MRI files and their segmentations.')
    parser.add_argument('--mri_dir', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a1_MRI_2D/train", help='Path to the directory containing MRI files.')
    parser.add_argument('--seg_dir', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a2_mri_Seg2D_label2/train", help='Path to the directory containing segmentation files.')
    parser.add_argument('--csv_file', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/training/label2/train.csv", help='Path to the output CSV file.')

    args = parser.parse_args()

    mri_files, seg_files = find_mri_and_seg_files(args.mri_dir, args.seg_dir)
    create_csv_referencing_mri_and_seg(mri_files, seg_files, args.csv_file)

if __name__ == "__main__":
    main()
