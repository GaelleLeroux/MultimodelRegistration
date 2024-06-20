import SimpleITK as sitk
import os
import argparse
import numpy as np


def MaskedImage(fixed_image_path, fixed_seg_path, folder_output, suffix, SegLabel=None):
    """Mask the fixed image with the fixed segmentation and write it to a file"""
    fixed_image_sitk = sitk.ReadImage(fixed_image_path)
    fixed_seg_sitk = sitk.ReadImage(fixed_seg_path)
    fixed_seg_sitk.SetOrigin(fixed_image_sitk.GetOrigin())
    fixed_image_masked = applyMask(fixed_image_sitk, fixed_seg_sitk, label=SegLabel)
    # Write the masked image
    
    base_name, ext = os.path.splitext(fixed_image_path)
    if base_name.endswith('.nii'):  # Case for .nii.gz
        ext = '.nii.gz'
    
    file_name = os.path.basename(fixed_image_path)
    file_name_without_ext = os.path.splitext(os.path.splitext(file_name)[0])[0]

    # Construction du chemin de fichier de sortie
    output_path = os.path.join(folder_output, f"{file_name_without_ext}_{suffix}{ext}")
            
    # Debugging folder_output
    sitk.WriteImage(sitk.Cast(fixed_image_masked, sitk.sitkInt16), output_path)
    

    return output_path


def applyMask(image, mask, label):
    """Apply a mask to an image."""
    # Cast the image to float32
    # image = sitk.Cast(image, sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(mask)
    if label is not None and label in np.unique(array):
        array = np.where(array == label, 1, 0)
        mask = sitk.GetImageFromArray(array)
        mask.CopyInformation(image)

    return sitk.Mask(image, mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invert the intensity of MRI images while keeping the background at 0.")
    parser.add_argument("--path_file", type=str, help="The path to the folder containing the MRI files", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/z0_test_folder/inverse_mri/A001_T2_mri_inv.nii.gz")
    parser.add_argument("--seg_file", type=str, help="The path to the folder containing the MRI files", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/z0_test_folder/mri/A001_T2_mri_seg_mand.nii.gz")
    parser.add_argument("--folder_output", type=str, help="The path to the output folder for the inverted files",default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/z0_test_folder/mask")
    parser.add_argument("--suffix", type=str, help="The suffix to add to the output filenames",default="mask")
    parser.add_argument("--seg_label", type=str, help="label of the segmentation",default=1)

    args = parser.parse_args()
    MaskedImage(args.path_file, args.seg_file, args.folder_output,args.suffix,args.seg_label)