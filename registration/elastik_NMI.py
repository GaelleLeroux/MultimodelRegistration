import os
import itk
import argparse
import SimpleITK as sitk
import numpy as np

def MatrixRetrieval(TransformParameterMapObject):
    """Retrieve the matrix from the transform parameter map"""
    ParameterMap = TransformParameterMapObject.GetParameterMap(0)

    if ParameterMap["Transform"][0] == "AffineTransform":
        matrix = [float(i) for i in ParameterMap["TransformParameters"]]
        # Convert to a sitk transform
        transform = sitk.AffineTransform(3)
        transform.SetParameters(matrix)

    elif ParameterMap["Transform"][0] == "EulerTransform":
        A = [float(i) for i in ParameterMap["TransformParameters"][0:3]]
        B = [float(i) for i in ParameterMap["TransformParameters"][3:6]]
        # Convert to a sitk transform
        transform = sitk.Euler3DTransform()
        transform.SetRotation(angleX=A[0], angleY=A[1], angleZ=A[2])
        transform.SetTranslation(B)

    return transform

def ComputeFinalMatrix(Transforms):
    """Compute the final matrix from the list of matrices and translations"""
    Rotation, Translation = [], []
    for i in range(len(Transforms)):
        Rotation.append(Transforms[i].GetMatrix())
        Translation.append(Transforms[i].GetTranslation())

    # Compute the final rotation matrix
    final_rotation = np.reshape(np.asarray(Rotation[0]), (3, 3))
    for i in range(1, len(Rotation)):
        final_rotation = final_rotation @ np.reshape(np.asarray(Rotation[i]), (3, 3))

    # Compute the final translation matrix
    final_translation = np.reshape(np.asarray(Translation[0]), (1, 3))
    for i in range(1, len(Translation)):
        final_translation = final_translation + np.reshape(
            np.asarray(Translation[i]), (1, 3)
        )

    # Create the final transform
    final_transform = sitk.Euler3DTransform()
    final_transform.SetMatrix(final_rotation.flatten().tolist())
    final_transform.SetTranslation(final_translation[0].tolist())

    return final_transform

def ElastixReg(fixed_image, moving_image, initial_transform=None, log_file_path=None):
    """Perform a registration using elastix with Normalized Mutual Information"""

    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)

    # ParameterMap
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap("rigid")

    # Modify the parameter map to use Normalized Mutual Information
    default_rigid_parameter_map["Metric"] = ["NormalizedMutualInformation"]
    default_rigid_parameter_map["Metric0Weight"] = ["1.0"]
    default_rigid_parameter_map["NumberOfHistogramBins"] = ["64"]
    default_rigid_parameter_map["UseNormalization"] = ["false"]
    default_rigid_parameter_map["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    default_rigid_parameter_map["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    default_rigid_parameter_map["AutomaticScalesEstimation"] = ["true"]
    default_rigid_parameter_map["AutomaticTransformInitialization"] = ["true"]

    parameter_object.AddParameterMap(default_rigid_parameter_map)
    parameter_object.SetParameter("ErodeMask", "true")
    parameter_object.SetParameter("WriteResultImage", "false")
    parameter_object.SetParameter("MaximumNumberOfIterations", "10000")
    parameter_object.SetParameter("NumberOfResolutions", "4")
    parameter_object.SetParameter("NumberOfSpatialSamples", "10000")

    elastix_object.SetParameterObject(parameter_object)
    if initial_transform is not None:
        elastix_object.SetInitialTransformParameterObject(initial_transform)

    # Additional parameters
    elastix_object.SetLogToConsole(False)
    if log_file_path:
        print("log_path : ", log_file_path)
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
                    Transforms = []
                    print(f"Registering CBCT: {cbct_path} with MRI: {mri_path}")
                    cbct_image = itk.imread(cbct_path, itk.F)
                    mri_image = itk.imread(mri_path, itk.F)
                    print(f"CBCT size: {cbct_image.GetLargestPossibleRegion().GetSize()}")
                    print(f"MRI size: {mri_image.GetLargestPossibleRegion().GetSize()}")
                    log_file_path = os.path.join(output_folder, f'{patient_id}_registration.log')

                    TransformObj_Fine = ElastixReg(cbct_image, mri_image, log_file_path=log_file_path)

                    transforms_Fine = MatrixRetrieval(TransformObj_Fine)
                    Transforms.append(transforms_Fine)
                    transform = ComputeFinalMatrix(Transforms)
                    output_transform_path = os.path.join(output_folder, f'{patient_id}_reg.tfm')
                    print("output_transform_path : ",output_transform_path)
                    sitk.WriteTransform(transform, output_transform_path)
                    
                    print(f"Saved transform to {output_transform_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
    parser.add_argument('--cbct_folder', type=str, help='Path to the folder containing CBCT images')
    parser.add_argument('--mri_folder', type=str, help='Path to the folder containing MRI images')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    
    args = parser.parse_args()
    main(args.cbct_folder, args.mri_folder, args.output_folder)
