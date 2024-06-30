import SimpleITK as sitk
import argparse
import os


def get_corresponding_file(folder, patient_id, modality):
    """Get the corresponding file for a given patient ID and modality."""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith(patient_id) and modality in file and file.endswith(".nii.gz"):
                return os.path.join(root, file)
    return None

def apply_transformation(input_image_path, transform_path, output_image_path):
    # Read the input image
    input_image = sitk.ReadImage(input_image_path)

    # Read the transformation
    transform = sitk.ReadTransform(transform_path)

    # Apply the transformation to the input image
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetReferenceImage(input_image)
    resample_filter.SetTransform(transform)
    resample_filter.SetInterpolator(sitk.sitkLinear)
    
    transformed_image = resample_filter.Execute(input_image)

    # Save the transformed image
    sitk.WriteImage(transformed_image, output_image_path)
    
def main(args):
    
    for transform in os.listdir(args.transform_folder):
        if transform.endswith(".tfm"):
            patient_id = transform.split("_MR")[0].split(".tfm")[0].split("_MA")[0]
            
            # input_file = get_corresponding_file(args.input_folder, patient_id, "_MR_")
            seg_file = get_corresponding_file(args.input_folder, patient_id, "_MR_")
            if type(seg_file)!=str:
                continue
            print("*"*100)
            print("input_folder : ",args.input_folder)
            print("patient_id : ",patient_id)
            # output_image_path = os.path.join(args.output_folder_file,os.path.basename(input_file).replace('.nii.gz', f'_reg.nii.gz'))
            output_seg_path = os.path.join(args.output_folder,os.path.basename(seg_file).replace('.nii.gz', f'_B.nii.gz'))
            
            transform = os.path.join(args.transform_folder,transform)
            
            # print("-"*150)
            # print("input_file : ",input_file)
            # print("transform : ",transform)
            # print("output_image_path : ",output_image_path)
            
            print("seg_file : ",seg_file)
            print("transform : ",transform)
            print("output_seg_path : ",output_seg_path)
            print("*"*100)
            # apply_transformation(input_file,transform,output_image_path)
            apply_transformation(seg_file,transform,output_seg_path)
            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply a transformation to an image.')

    # parser.add_argument("--input_folder", type=str, help="Path to the input image file (e.g., .nii.gz)", default="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Registration/transform/")
    parser.add_argument("--input_folder", type=str, help="Path to the input image file (e.g., .nii.gz)",default="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Registration/MRI_seg")
    parser.add_argument("--transform_folder", type=str, help="Path to the transformation file (e.g., .tfm)",default="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Registration/transform/")
    # parser.add_argument("--output_folder_file", type=str, help="Path to save the output transformed image (e.g., .nii.gz)",default="/home/lucia/Documents/Gaelle/Data/MultimodelReg/Registration/not_worked/reg/mri")
    parser.add_argument("--output_folder", type=str, help="Path to save the output transformed image (e.g., .nii.gz)", default="/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Registration/MRI_seg_transform/")

    args = parser.parse_args()

    main(args)
