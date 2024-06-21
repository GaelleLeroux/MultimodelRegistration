import argparse
import os
import itk
import SimpleITK as sitk
from AREG_CBCT_utils.utils import ElastixApprox, MatrixRetrieval, ElastixReg, ComputeFinalMatrix, ResampleImage
import numpy as np

def get_corresponding_file(folder, patient_id, modality):
    """Get the corresponding file for a given patient ID and modality."""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith(patient_id) and modality in file and file.endswith(".nii.gz"):
                return os.path.join(root, file)
    return None

def main(args):
    cbct_folder = args.cbct_folder
    mri_folder = args.mri_folder
    cbct_mask_folder = args.cbct_mask_folder
    mri_mask_folder = args.mri_mask_folder
    cbct_seg_folder = args.cbct_seg_folder
    mri_seg_folder = args.mri_seg_folder
    output_folder = args.output_folder
    mri_original_folder = args.mri_original_folder

    for cbct_file in os.listdir(cbct_folder):
        if cbct_file.endswith(".nii.gz") and "_CBCT_" in cbct_file:
            mri_mask_path="None"
            patient_id = cbct_file.split("_CBCT_")[0]
            
            cbct_path = os.path.join(cbct_folder, cbct_file)
            mri_path = get_corresponding_file(mri_folder, patient_id, "_MR_")
            mri_path_original = get_corresponding_file(mri_original_folder, patient_id, "_MR_")
            
            if not mri_path:
                print(f"Corresponding MRI file for {cbct_file} not found.")
                continue

            cbct_mask_path = get_corresponding_file(cbct_mask_folder, patient_id, "_CBCT_")
            if mri_mask_folder!="None" : 
                mri_mask_path = get_corresponding_file(mri_mask_folder, patient_id, "_MR_")

            cbct_seg_path = get_corresponding_file(cbct_seg_folder, patient_id, "_CBCT_")
            mri_seg_path = get_corresponding_file(mri_seg_folder, patient_id, "_MR_")

            if not all([cbct_mask_path, mri_mask_path, cbct_seg_path, mri_seg_path]):
                print(f"One or more corresponding mask or segmentation files for {cbct_file} not found.")
                continue
            # print("cbct_mask_path : ",cbct_mask_path)
            # print("mri_path : ",mri_path)

            process_images(cbct_path, mri_path, cbct_mask_path, mri_mask_path, cbct_seg_path, mri_seg_path, output_folder,patient_id,mri_path_original)

def process_images(fixed_image_path, moving_image_path, fixed_mask_path, moving_mask_path, fixed_seg_path, moving_seg_path, output_folder, patient_id,mri_path_original):
    
    # print("moving_image_path : ",moving_image_path)
    # print("fixed_mask_path : ",fixed_mask_path)
    fixed_image = itk.imread(fixed_image_path, itk.F)
    moving_image = itk.imread(moving_image_path, itk.F)
    fixed_image_mask = itk.imread(fixed_mask_path, itk.F)
    # print_min_max(moving_image, "Moving Image")
    # print_min_max(fixed_image_mask, "Fixed Image Mask")
    # moving_image_mask = itk.imread(moving_mask_path, itk.F)
    # fixed_seg = itk.imread(fixed_seg_path, itk.F)
    # moving_seg = itk.imread(moving_seg_path, itk.F)
    Transforms = []
    
    # TransformObj_Approx = np.eye(4)  # 4x4 identity matrix
    try : 
        TransformObj_Fine = ElastixReg(fixed_image_mask, moving_image, initial_transform=None)
    except Exception as e:
        print("An error occurred while reading the images or during the registration process:")
        print(e)
        print(f"{patient_id} failed")
        return
    
    print(f"{patient_id} a ete traite")
    transforms_Fine = MatrixRetrieval(TransformObj_Fine)
    Transforms.append(transforms_Fine)
    transform = ComputeFinalMatrix(Transforms)
    resample_t2 = sitk.Cast(ResampleImage(sitk.ReadImage(mri_path_original), transform), sitk.sitkInt16)
    
    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, f"{patient_id}_reg.nii.gz")
    sitk.WriteImage(resample_t2, output_image_path)
    sitk.WriteTransform(transform, os.path.join(output_folder, f"{patient_id}_reg.tfm"))
    

def print_min_max(image, image_name):
    image_array = itk.array_from_image(image)
    min_value = image_array.min()
    max_value = image_array.max()
    print(f"{image_name} - Min: {min_value}, Max: {max_value}")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AREG MRI folder')

    parser.add_argument("--cbct_folder", type=str,  help="Folder containing CBCT images.", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/b2_folder_CBCT/test")
    parser.add_argument("--cbct_mask_folder", type=str, help="Folder containing CBCT masks.", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/b5_folder_CBCT_norm_mask/test_percentile=[10,95]_norm=[0,3]")
    parser.add_argument("--cbct_seg_folder", type=str,  help="Folder containing CBCT segmentations.", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/b2_folder_CBCT_l2/test")
    parser.add_argument("--mri_folder", type=str,  help="Folder containing MRI images.", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/a4_folder_MRI_invert_norm/a03_test_percentile=[0,100]_norm=[0,4]")
    parser.add_argument("--mri_original_folder", type=str,  help="Folder containing original MRI.", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/a2_folder_MRI/test")
    parser.add_argument("--mri_mask_folder", type=str,  help="Folder containing MRI masks.", default="None")
    parser.add_argument("--mri_seg_folder", type=str,  help="Folder containing MRI segmentations.", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/a2_folder_MRI_seg_l2/test")
    parser.add_argument("--output_folder", type=str,  help="Folder to save the output files.",default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/z1_test/a03_mri:inv+norm[0,4]+p[0,100]_cbct:norm[0,3]+p[10,95]+mask")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
    main(args)
