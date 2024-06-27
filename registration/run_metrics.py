import os
import argparse
import pandas as pd
from metrics import NCC, NMI
from torch import Tensor
import SimpleITK as sitk
import torch
from ssim import SSIM

def load_nifti_as_tensor(file_path: str) -> Tensor:
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)
    tensor = torch.tensor(array, dtype=torch.float32)
    tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

def get_corresponding_file(folder, patient_id, modality):
    """Get the corresponding file for a given patient ID and modality."""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith(patient_id) and modality in file and file.endswith(".nii.gz"):
                return os.path.join(root, file)
    return None

def compute_metric(ncc_loss,fixed,warped,mask):
    if mask==None:
        loss = ncc_loss(fixed, warped)
    else :
        loss = ncc_loss(fixed, warped,mask)
    print("Loss without mask:", loss.item())
    return loss.item()

def main(input_folder_cbct, input_folder_mri, output_folder, name_csv,seg_folder):
    data = []
    metric = NCC(use_mask=False)
    metric = NMI(use_mask=False)
    metric = SSIM()
    if seg_folder!="None":
            metric = NMI(use_mask=True)

    for cbct_file in os.listdir(input_folder_cbct):
        if cbct_file.endswith(".nii.gz") and "_CBCT_" in cbct_file:
            patient_id = cbct_file.split("_CBCT_")[0]
            mask = None
            
            cbct_path = os.path.join(input_folder_cbct, cbct_file)
            mri_path = get_corresponding_file(input_folder_mri, patient_id, "_MR_")
            if seg_folder!="None":
                seg_path = get_corresponding_file(seg_folder, patient_id, "_CBCT_")
                mask = load_nifti_as_tensor(seg_path).bool()
            print("*"*100)
            print("cbct_path : ",cbct_path)
            print("mri_path : ",mri_path)

            fixed = load_nifti_as_tensor(cbct_path)
            warped = load_nifti_as_tensor(mri_path)
            print("size fixed : ",fixed.shape)
            
            if seg_folder!="None":
                loss = compute_metric(metric,fixed,warped,mask)
            else :
                loss = compute_metric(metric,fixed,warped,mask)
            
            data.append([patient_id, loss])

    df = pd.DataFrame(data, columns=["Patient ID", "Loss"])
    
    output_csv_path = os.path.join(output_folder, f"{name_csv}.csv")
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link CBCT files with their corresponding MRI files and save their sizes to a CSV.")
    parser.add_argument("--input_folder_cbct", type=str, help="Input folder containing CBCT NIfTI files")
    parser.add_argument("--input_folder_cbct_seg", type=str, help="Input folder containing CBCT NIfTI files",default="None")
    parser.add_argument("--input_folder_mri", type=str, help="Input folder containing MRI NIfTI files")
    parser.add_argument("--output_folder", type=str, help="Output folder to save the CSV file")
    parser.add_argument("--name_csv", type=str, help="Name of the output CSV file (without extension)")

    args = parser.parse_args()
    main(args.input_folder_cbct, args.input_folder_mri, args.output_folder, args.name_csv,args.input_folder_cbct_seg)
