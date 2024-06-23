import argparse
import os
import itk
import SimpleITK as sitk
from AREG_CBCT_utils.utils import ElastixApprox, MatrixRetrieval, ComputeFinalMatrix, ResampleImage
import numpy as np

def ElastixReg(fixed_image, moving_image, initial_transform=None):
    """Perform a registration using elastix with a rigid transform and possibly an initial transform"""

    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    # elastix_object.SetFixedMask(fixed_mask)

    # ParameterMap
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap("rigid")
    parameter_object.AddParameterMap(default_rigid_parameter_map)
    parameter_object.SetParameter("ErodeMask", "true")
    parameter_object.SetParameter("WriteResultImage", "false")
    parameter_object.SetParameter("MaximumNumberOfIterations", "10000")
    
    parameter_object.SetParameter("NumberOfResolutions", "3")  # Use multi-resolution strategy
    parameter_object.SetParameter("NumberOfSpatialSamples", "1000")  # Further reduce the number of spatial samples

    # Additional parameters to handle out-of-bounds sampling
    parameter_object.SetParameter("CheckNumberOfSamples", "true")
    parameter_object.SetParameter("MaximumNumberOfSamplingAttempts", "100")
    print(1)

    elastix_object.SetParameterObject(parameter_object)
    if initial_transform is not None:
        elastix_object.SetInitialTransformParameterObject(initial_transform)

    print(2)
    # Additional parameters
    elastix_object.SetLogToConsole(True)
    
    print(3)

    # Execute registration
    elastix_object.UpdateLargestPossibleRegion()
    print(4)

    TransParamObj = elastix_object.GetTransformParameterObject()
    print(5)

    return TransParamObj

def process_images(mri_path, cbct_mask_path,mri_path_original,patient_id,output_folder):
    

    # cbct_path = itk.imread(cbct_path, itk.F)
    try : 
        mri_path = itk.imread(mri_path, itk.F)
        cbct_mask_path = itk.imread(cbct_mask_path, itk.F)
    except KeyError as e:
        print("An error occurred while reading the images or during the registration process:")
        print(e)
        print(f"{patient_id} failed")
        return

    Transforms = []
    
    # TransformObj_Approx = np.eye(4)  # 4x4 identity matrix
    try : 
        TransformObj_Fine = ElastixReg(cbct_mask_path, mri_path, initial_transform=None)
    except Exception as e:
        print("An error occurred while reading the images or during the registration process:")
        print(e)
        print(f"{patient_id} failed")
        return
    
    print(f"{patient_id} a ete traite")
    transforms_Fine = MatrixRetrieval(TransformObj_Fine)
    Transforms.append(transforms_Fine)
    transform = ComputeFinalMatrix(Transforms)
    
    os.makedirs(output_folder, exist_ok=True)
    
    output_image_path = os.path.join(output_folder,os.path.basename(mri_path_original).replace('.nii.gz', f'_reg.nii.gz'))
    output_image_path_transform = os.path.join(output_folder,os.path.basename(mri_path_original).replace('.nii.gz', f'_reg_transform.tfm'))
    
    sitk.WriteTransform(transform, output_image_path_transform)
    
    resample_t2 = sitk.Cast(ResampleImage(sitk.ReadImage(mri_path_original), transform), sitk.sitkInt16)
    sitk.WriteImage(resample_t2, output_image_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AREG MRI folder')

    parser.add_argument("--mri_path", type=str,  help="MRI file to use for registration", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a5_folder_14_more_data/a2_MRI_inv_norm/test_percentile=[0,100]_norm=[0,100]/B012_MR_CropCo_inv_percentile=[0,100]_norm=[0,100].nii.gz")
    parser.add_argument("--cbct_mask_path", type=str,  help="CBCT mask file to use for registration", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a5_folder_14_more_data/b3_CBCT_inv_norm_mask:l2/test_percentile=[10,95]_norm=[0,75]/B012_CBCT_CropCo_percentile=[10,95]_norm=[0,75]_mask.nii.gz")
    parser.add_argument("--mri_path_original", type=str, help="Original MRI without preprocess", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a5_folder_14_more_data/a0_MRI/B012_MR_CropCo.nii.gz")
    parser.add_argument("--patient_id", type=str,  help="Name of patient", default="B012")
    
    parser.add_argument("--output_folder", type=str,  help="Folder to save the output files.",default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/a5_folder_14_more_data/z01_output/a01_mri:inv+norm[0,100]+p[0,100]_cbct:norm[0,75]+p[10,95]+mask")

    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
    process_images(args.mri_path,args.cbct_mask_path,args.mri_path_original,args.patient_id,args.output_folder)
