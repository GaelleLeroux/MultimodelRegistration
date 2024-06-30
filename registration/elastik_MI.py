import os
import itk
import argparse

def ElastixReg(fixed_image, moving_image, initial_transform=None, log_file_path=None):
    """Perform a registration using elastix with AdvancedMattestMtutalInformation"""

    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    # elastix_object.SetFixedMask(fixed_mask)

    # ParameterMap
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap("rigid")

    # Modify the parameter map to use AdvancedMattestMtutalInformation
    default_rigid_parameter_map["Metric"] = ["AdvancedMattestMtutalInformation"]
    default_rigid_parameter_map["Metric0Weight"] = ["1.0"]
    default_rigid_parameter_map["NumberOfHistogramBins"] = ["32"]
    default_rigid_parameter_map["UseNormalization"] = ["false"]
    default_rigid_parameter_map["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    default_rigid_parameter_map["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]

    parameter_object.AddParameterMap(default_rigid_parameter_map)
    parameter_object.SetParameter("ErodeMask", "true")
    parameter_object.SetParameter("WriteResultImage", "false")
    parameter_object.SetParameter("MaximumNumberOfIterations", "10000")
    parameter_object.SetParameter("NumberOfResolutions", "1")
    parameter_object.SetParameter("NumberOfSpatialSamples", "10000")
    # parameter_object.SetParameter("MaximumNumberOfSamplingAttempts", "25")

    elastix_object.SetParameterObject(parameter_object)
    if initial_transform is not None:
        elastix_object.SetInitialTransformParameterObject(initial_transform)

    # Additional parameters
    elastix_object.SetLogToConsole(False)
    if log_file_path:
        print("log_path : ",log_file_path)
        log_directory = os.path.dirname(log_file_path)
        elastix_object.LogToFileOn()
        elastix_object.SetLogFileName(log_file_path)
        elastix_object.SetOutputDirectory(log_directory)
        print("pass log file")
    else:
        elastix_object.LogToConsoleOn()

    # Execute registration
    elastix_object.UpdateLargestPossibleRegion()

    TransParamObj = elastix_object.GetTransformParameterObject()

    return TransParamObj

def get_corresponding_file(folder, patient_id, modality):
    """Get the corresponding file for a given patient ID and modality."""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith(patient_id) and modality in file and file.endswith(".nii.gz"):
                return os.path.join(root, file)
    return None

def main(cbct_folder, mri_folder, output_folder):
    for root, _, files in os.walk(cbct_folder):
        for cbct_file in files:
            if "_CBCT_" in cbct_file and cbct_file.endswith(".nii.gz"):
                patient_id = cbct_file.split("_CBCT_")[0]
                cbct_path = os.path.join(root, cbct_file)
                mri_path = get_corresponding_file(mri_folder, patient_id, "_MR_")
                

                if mri_path:
                    print(f"Registering CBCT: {cbct_path} with MRI: {mri_path}")
                    cbct_image = itk.imread(cbct_path, itk.F)
                    mri_image = itk.imread(mri_path, itk.F)
                    print(f"CBCT size: {cbct_image.GetLargestPossibleRegion().GetSize()}")
                    print(f"MRI size: {mri_image.GetLargestPossibleRegion().GetSize()}")
                    log_file_path = os.path.join(output_folder, f'{patient_id}_registration.log')
                    transform = ElastixReg(cbct_image, mri_image, log_file_path=log_file_path)
                    
                    output_transform_path = os.path.join(output_folder, f'{patient_id}_reg.tfm')
                    itk.transformwrite(transform, output_transform_path)
                    print(f"Saved transform to {output_transform_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
    parser.add_argument('--cbct_folder', type=str, help='Path to the folder containing CBCT images')
    parser.add_argument('--mri_folder', type=str, help='Path to the folder containing MRI images')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    
    args = parser.parse_args()
    main(args.cbct_folder, args.mri_folder, args.output_folder)
