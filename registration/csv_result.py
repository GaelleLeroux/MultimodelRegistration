import os
import argparse
import SimpleITK as sitk
import numpy as np
import pandas as pd

def get_corresponding_file(folder, patient_id, modality):
    """Get the corresponding file for a given patient ID and modality."""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith(patient_id) and modality in file and file.endswith(".nii.gz"):
                return os.path.join(root, file)
    return None

def compute_nmi(fixed_image, moving_image):
    """
    Compute the Normalized Mutual Information between two images using SimpleITK's ImageRegistrationMethod.
    """
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1)
    registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY))
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.Execute(fixed_image, moving_image)
    return registration_method.MetricEvaluate(fixed_image, moving_image)

def compute_dice_coefficient(seg1, seg2):
    """
    Compute the Dice coefficient between two binary segmentations.
    """
    intersection = np.logical_and(seg1, seg2)
    dice = 2. * intersection.sum() / (seg1.sum() + seg2.sum())
    return dice

def main(args):
    results = []

    for root, _, files in os.walk(args.input_folder_cbct):
        for file in files:
            if file.endswith(".nii.gz") and "_CBCT_" in file:
                cbct_file_path = os.path.join(root, file)
                patient_id = file.split("_CBCT_")[0]
                mri_path = get_corresponding_file(args.input_folder_mri, patient_id, "_MR_")
                if mri_path:
                    # Load images
                    fixed_image = sitk.ReadImage(mri_path, sitk.sitkFloat32)
                    moving_image = sitk.ReadImage(cbct_file_path, sitk.sitkFloat32)
                    
                    # Compute NMI
                    nmi_value = compute_nmi(fixed_image, moving_image)
                    
                    # Load segmentation masks
                    fixed_seg_path = get_corresponding_file(args.input_folder_mri, patient_id, "_MR_seg_")
                    moving_seg_path = get_corresponding_file(args.input_folder_cbct, patient_id, "_CBCT_seg_")
                    if fixed_seg_path and moving_seg_path:
                        fixed_seg = sitk.GetArrayFromImage(sitk.ReadImage(fixed_seg_path, sitk.sitkUInt8))
                        moving_seg = sitk.GetArrayFromImage(sitk.ReadImage(moving_seg_path, sitk.sitkUInt8))
                        
                        # Ensure the segmentations are binary
                        fixed_seg = fixed_seg > 0
                        moving_seg = moving_seg > 0
                        
                        # Compute Dice coefficient
                        dice_value = compute_dice_coefficient(fixed_seg, moving_seg)
                    else:
                        print("error dice computation")
                        dice_value = None
                    
                    results.append([patient_id, nmi_value, dice_value])
    
    # Create a DataFrame and save to CSV
    results_df = pd.DataFrame(results, columns=['PatientID', 'NMI', 'Dice'])
    output_csv_path = os.path.join(args.output_folder, args.name_csv)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MRI-CBCT registration quality.")
    parser.add_argument('--input_folder_mri', help='Input folder for MRI files', default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration_data_closer/z0_reg/a01_mri:inv+norm[0,100]+p[0,100]_cbct:norm[0,75]+p[10,95]+mask")
    parser.add_argument('--input_folder_mri_seg', help='Input folder segmentation for MRI files', default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration_data_closer/z0_reg/a01_mri:inv+norm[0,100]+p[0,100]_cbct:norm[0,75]+p[10,95]+mask")
    
    parser.add_argument('--input_folder_cbct', help='Input folder for CBCT files',default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration_data_closer/b0_CBCT")
    parser.add_argument('--input_folder_cbct_seg', help='Input folder segmentation for CBCT files', default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration_data_closer/z0_reg/a01_mri:inv+norm[0,100]+p[0,100]_cbct:norm[0,75]+p[10,95]+mask")
    
    parser.add_argument('--output_folder', help='Output folder for the results CSV',default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration_data_closer/z0_reg")
    parser.add_argument('--name_csv', help='Name of the output CSV file',default='a01_result.csv')

    args = parser.parse_args()
    main(args)
