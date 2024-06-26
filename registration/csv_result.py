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

    for root, _, files in os.walk(args.input_folder_cbct_seg):
        for file in files:
            if file.endswith(".nii.gz") and "_CBCT_" in file:
                cbct_seg_path= os.path.join(root, file)
                patient_id = file.split("_CBCT_")[0]
                
                mri_seg_path = get_corresponding_file(args.input_folder_mri_seg, patient_id, "_MR_")
                mri_transform_seg_path = get_corresponding_file(args.input_folder_mri_seg_after, patient_id, "_MR_")
                if mri_seg_path and mri_transform_seg_path:
                    try : 
                        cbct_seg = sitk.GetArrayFromImage(sitk.ReadImage(cbct_seg_path, sitk.sitkUInt8))
                        mri_seg_seg = sitk.GetArrayFromImage(sitk.ReadImage(mri_seg_path, sitk.sitkUInt8))
                        mri_transform_seg = sitk.GetArrayFromImage(sitk.ReadImage(mri_transform_seg_path, sitk.sitkUInt8))
                        
                        # Ensure the segmentations are binary
                        cbct_seg = cbct_seg > 0
                        mri_seg_seg = mri_seg_seg > 0
                        mri_transform_seg = mri_transform_seg > 0
                        
                        # Compute Dice coefficient
                        dice_value_before = compute_dice_coefficient(cbct_seg, mri_seg_seg)
                        dice_value_after = compute_dice_coefficient(cbct_seg, mri_transform_seg)
                        results.append([patient_id, dice_value_before,dice_value_after])
                    except KeyError as e :
                        print(e)
                else:
                    print("*"*100)
                    print("cbct_seg_path : ",cbct_seg_path)
                    print("mri_seg_path : ",mri_seg_path)
                    print("mri_transform_seg_path : ",mri_transform_seg_path)
                    print("error dice computation")
                    # dice_value = None
                    
                    
    
    # Create a DataFrame and save to CSV
    results_df = pd.DataFrame(results, columns=['PatientID', 'dice_value_before','dice_value_after'])
    output_csv_path = os.path.join(args.output_folder, args.name_csv)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MRI-CBCT registration quality.")

    parser.add_argument('--input_folder_mri_seg', help='Input folder segmentation for MRI files', default="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Registration/MRI_seg")
    parser.add_argument('--input_folder_mri_seg_after', help='Input folder segmentation for MRI files', default="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Registration/MRI_seg_transform/")
    
    parser.add_argument('--input_folder_cbct_seg', help='Input folder segmentation for CBCT files', default="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Registration/CBCT_seg_resample/")
    
    parser.add_argument('--output_folder', help='Output folder for the results CSV',default="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Registration/")
    parser.add_argument('--name_csv', help='Name of the output CSV file',default='dice_comparaison.csv')

    args = parser.parse_args()
    main(args)
    