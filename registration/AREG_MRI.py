import argparse
import os
import itk
import SimpleITK as sitk
from AREG_CBCT_utils.utils import ElastixApprox,MatrixRetrieval,ElastixReg,ComputeFinalMatrix,ResampleImage


def main(args):
    (
        fixed_image_path,
        moving_image_path,
        fixed_mask_path,
        moving_mask_path,
        fixed_seg_path,
        moving_seg_path,
        output_folder
    ) = (
        args.cbct,
        args.mri,
        args.cbct_mask,
        args.mri_mask,
        args.cbct_seg,
        args.mri_seg,
        args.output_folder
    )
    
    fixed_image = itk.imread(fixed_image_path, itk.F)
    moving_image = itk.imread(moving_image_path, itk.F)
    fixed_image_mask = itk.imread(fixed_mask_path, itk.F)
    # moving_image_mask = itk.imread(moving_mask_path, itk.F)
    # fixed_seg = itk.imread(fixed_seg_path, itk.F)
    # moving_seg = itk.imread(moving_seg_path, itk.F)
    Transforms = []
    
    ###############################################################################################
    TransformObj_Approx = ElastixApprox(fixed_image, moving_image) ### approximation betwwen cbct and mri, full scan
    transforms_Approx = MatrixRetrieval(TransformObj_Approx)
    Transforms.append(transforms_Approx)
    transform = ComputeFinalMatrix(Transforms)
    sitk.WriteTransform(transform, os.path.join(output_folder,"normal.tfm"))
    resample_t2 = sitk.Cast(
        ResampleImage(sitk.ReadImage(moving_image_path), transform), sitk.sitkInt16
    )
    sitk.WriteImage(resample_t2, os.path.join(output_folder,"normal.nii.gz"))
    ####
    ############################################################################################### register between fix cbct mask and full mri
    # TransformObj_Approx=None
    TransformObj_Fine = ElastixReg( 
        fixed_image_mask, moving_image, initial_transform=TransformObj_Approx
    )
    
    transforms_Fine = MatrixRetrieval(TransformObj_Fine)
    Transforms.append(transforms_Fine)
    transform = ComputeFinalMatrix(Transforms)
    print("moving_image_path : ",moving_image_path)
    print("moving_image_path : ",moving_image_path)
    resample_t2 = sitk.Cast(
        ResampleImage(sitk.ReadImage(moving_image_path), transform), sitk.sitkInt16
    )
    
    sitk.WriteImage(resample_t2, os.path.join(output_folder,"normal_reg.nii.gz"))
    sitk.WriteTransform(transform, os.path.join(output_folder,"normal_reg.tfm"))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AREG CBCT')

    parser.add_argument("--cbct", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/b2_folder_CBCT/test/A001_CBCT_Crop.nii.gz")
    parser.add_argument("--cbct_mask", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/b5_folder_CBCT_norm_mask/test/A001_CBCT_Crop_norm_mask.nii.gz")
    parser.add_argument("--cbct_seg", default="None")
    
    parser.add_argument("--mri", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/a4_folder_MRI_invert_norm/test/A001_MR_Crop_inv_norm.nii.gz")
    parser.add_argument("--mri_mask", default="None")
    parser.add_argument("--mri_seg", default="None")
    
    parser.add_argument("--output_folder", default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Segmentation/Registration/z1_test/a01_invert_mri+norm_cbct_mri")
    # parser.add_argument('reg_lm',nargs=1)

    args = parser.parse_args()

    main(args)
