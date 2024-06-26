import SimpleITK as sitk
import numpy as np

def read_tfm(file_path):
    # Read the transformation from the file
    transform = sitk.ReadTransform(file_path)
    
    # Get the transformation matrix and offset
    matrix = np.array(transform.GetMatrix()).reshape((3, 3))
    offset = np.array(transform.GetTranslation())
    
    # Construct the 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = matrix
    transformation_matrix[:3, 3] = offset
    
    return transformation_matrix

def convert_lps_to_ras(matrix):
    reflection_matrix = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    transformed_matrix = reflection_matrix @ matrix @ reflection_matrix
    return transformed_matrix

def print_transformation_matrix(matrix):
    print("Transformation Matrix:")
    print(matrix)

if __name__ == "__main__":
    tfm_file = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Registration/B11_B12/z01_output/a01_mri:inv+norm[0,100]+p[10,95]_cbct:norm[0,100]+p[10,95]+mask/B011_MR_CropCo_reg_transform.tfm'
    transformation_matrix = read_tfm(tfm_file)
    print("Original Transformation Matrix (LPS):")
    print_transformation_matrix(transformation_matrix)
    
    ras_transformation_matrix = convert_lps_to_ras(transformation_matrix)
    print("Converted Transformation Matrix (RAS):")
    print_transformation_matrix(ras_transformation_matrix)



