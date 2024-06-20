import os
import pandas as pd
import argparse
import re

def find_mri_and_seg_files(mri_dir, seg_dir):
    mri_files = []
    seg_files = []

    # Regular expression to match the file pattern
    mri_pattern = re.compile(r"(.*)_MR_Crop.nii.gz$")
    seg_pattern1 = re.compile(r"(.*)_MR_Seg_clean_label1.nii.gz$")
    seg_pattern2 = re.compile(r"(.*)_MR_Seg_retouched_label1.nii.gz$")
    seg_pattern3 = re.compile(r"(.*)_MR_Seg_label1.nii.gz$")

    # Traverse the MRI directory
    for root, dirs, files in os.walk(mri_dir):
        for file in files:
            if mri_pattern.search(file):
                mri_files.append(os.path.join(root, file))

    # Traverse the segmentation directory
    for root, dirs, files in os.walk(seg_dir):
        for file in files:
            if seg_pattern1.search(file) or seg_pattern2.search(file) or seg_pattern3.search(file):
                seg_files.append(os.path.join(root, file))

    return mri_files, seg_files

def create_csv_referencing_mri_and_seg(mri_files, seg_files, csv_file):
    data = {
        "mri": [],
        "seg": []
    }

    
    for mri in mri_files:
        match = re.search(r"(.*)_MR_Crop.nii.gz$", os.path.basename(mri))
        if match:
            patient_name = match.group(1)
            seg_pattern1 = f"{patient_name}_MR_Seg_clean_label1.nii.gz"
            seg_pattern2 = f"{patient_name}_MR_Seg_retouched_label1.nii.gz"
            seg_pattern3 = f"{patient_name}_MR_Seg_label1.nii.gz"
            seg = next((s for s in seg_files if os.path.basename(s) == seg_pattern1 or os.path.basename(s) == seg_pattern2 or os.path.basename(s) == seg_pattern3), None)
        
            if seg:
                data["mri"].append(mri)
                data["seg"].append(seg)
                
    print("data : ",data)
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"CSV file created: {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='Create a CSV referencing MRI files and their segmentations.')
    parser.add_argument('--mri_dir', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/3D_more_data/a2_folder_MRI/valid", help='Path to the directory containing MRI files.')
    parser.add_argument('--seg_dir', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/3D_more_data/a2_folder_MRI_seg_l1/valid", help='Path to the directory containing segmentation files.')
    parser.add_argument('--csv_file', type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/3D_more_data/a3_csv_mri_l1/valid.csv", help='Path to the output CSV file.')

    args = parser.parse_args()

    mri_files, seg_files = find_mri_and_seg_files(args.mri_dir, args.seg_dir)
    print("mri_files : ",mri_files)
    print("seg_files : ",seg_files)
    create_csv_referencing_mri_and_seg(mri_files, seg_files, args.csv_file)

if __name__ == "__main__":
    main()
