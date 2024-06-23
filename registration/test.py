import SimpleITK as sitk
import argparse

def compute_mutual_information(fixed_image, moving_image):
    """Compute the Mutual Information metric between fixed and moving images."""
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Set up for metric evaluation without performing actual registration
    registration_method.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))
    registration_method.SetFixedImage(fixed_image)
    registration_method.SetMovingImage(moving_image)
    registration_method.Initialize()

    # Get the metric value
    metric_value = registration_method.MetricEvaluate(fixed_image, moving_image)
    return metric_value

def compute_normalized_cross_correlation(fixed_image, moving_image):
    """Compute the Normalized Cross-Correlation metric between fixed and moving images."""
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsNormalizedCorrelation()
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Set up for metric evaluation without performing actual registration
    registration_method.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))
    registration_method.SetFixedImage(fixed_image)
    registration_method.SetMovingImage(moving_image)
    registration_method.Initialize()

    # Get the metric value
    metric_value = registration_method.MetricEvaluate(fixed_image, moving_image)
    return metric_value

def apply_transformation(input_image, transform):
    """Apply a transformation to an input image."""
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetReferenceImage(input_image)
    resample_filter.SetTransform(transform)
    resample_filter.SetInterpolator(sitk.sitkLinear)
    
    transformed_image = resample_filter.Execute(input_image)
    return transformed_image

def main(fixed_image_path, moving_image_path, transform_path, output_image_path):
    # Read the images
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    
    # Compute metrics before registration
    mi_before = compute_mutual_information(fixed_image, moving_image)
    ncc_before = compute_normalized_cross_correlation(fixed_image, moving_image)
    print(f"Metrics before registration: MI={mi_before}, NCC={ncc_before}")
    
    # Read the transformation
    transform = sitk.ReadTransform(transform_path)

    # Apply the transformation to the moving image
    transformed_image = apply_transformation(moving_image, transform)
    
    # Save the transformed image
    sitk.WriteImage(transformed_image, output_image_path)
    
    # Compute metrics after registration
    mi_after = compute_mutual_information(fixed_image, transformed_image)
    ncc_after = compute_normalized_cross_correlation(fixed_image, transformed_image)
    print(f"Metrics after registration: MI={mi_after}, NCC={ncc_after}")

    # Compare metrics to assess improvement
    mi_improvement = mi_after > mi_before
    ncc_improvement = ncc_after > ncc_before
    print(f"Improvement: MI={mi_improvement}, NCC={ncc_improvement}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assess the quality of MRI and CBCT registration.')

    parser.add_argument("--fixed_image", type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Registration/worked/reg/cbct/A007_CBCT_Crop_MA.nii.gz", help="Path to the fixed image file (e.g., MRI.nii.gz)")
    parser.add_argument("--moving_image", type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Registration/worked/not_reg/mri/A007_MR_Crop.nii.gz", help="Path to the moving image file (e.g., CBCT.nii.gz)")
    parser.add_argument("--transform", type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Registration/worked/transform/A007_MR_Crop_reg_transform.tfm", help="Path to the transformation file (e.g., transform.tfm)")
    parser.add_argument("--output_image", type=str, default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Registration/worked", help="Path to save the output transformed image (e.g., registered_CBCT.nii.gz)")

    args = parser.parse_args()

    main(args.fixed_image, args.moving_image, args.transform, args.output_image)
