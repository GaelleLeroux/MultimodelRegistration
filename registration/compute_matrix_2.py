import os
import argparse
import SimpleITK as sitk
import pandas as pd
import numpy as np

def round_up(value, decimals=2):
    multiplier = 10 ** decimals
    return np.ceil(value * multiplier) / multiplier

def read_transform_file(file_path):
    transform = sitk.ReadTransform(file_path)
    print("transform :", transform)
    parameters = [round(p, 4) for p in transform.GetParameters()]
    fixed_parameters = [round(fp, 4) for fp in transform.GetFixedParameters()]
    print("file_path : ",file_path)
    print("(transform.GetParameters()) : ",    print(transform.GetParameters()))
    print("parameters : ",parameters)
    print("fixed_parameters : ",fixed_parameters)
    if isinstance(transform, sitk.Euler3DTransform):
        print(f"Transformation type: {type(transform)}")

        # Obtenir la matrice de rotation
        matrix = transform.GetMatrix()
        rotation_matrix = np.array(matrix).reshape((3, 3))
        rotation_matrix = round_up(rotation_matrix)

        # Obtenir le décalage (translation)
        offset = np.array(transform.GetTranslation())
        offset = round_up(offset)

        # Créer une matrice 4x4 pour la rotation/translation
        transformation_matrix = np.eye(4)  # Initialise avec la matrice identité 4x4
        transformation_matrix[:3, :3] = rotation_matrix  # Insère la matrice de rotation
        transformation_matrix[:3, 3] = offset  # Insère l'offset (translation) en quatrième colonne
        print("transformation_matrix : ",)

        return transformation_matrix
    else:
        raise ValueError("La transformation n'est pas de type Euler3DTransform.")
    
    # Initialize a 4x4 identity matrix
    matrix = np.eye(4)
    
    # if len(parameters) == 12:
    #     # 3D transformation matrix
    #     matrix[:3, :3] = np.array(parameters[:9]).reshape((3, 3))
    #     matrix[:3, 3] = np.array(parameters[9:12])
    # elif len(parameters) == 6:
    #     # 2D transformation matrix (extend to 4x4)
    #     # matrix[:2, :3] = np.array(parameters).reshape((2, 3))
    #     # matrix[2, 2] = 1
    #     # matrix[3, 3] = 1

    #     matrix[0, 1] = parameters[2]
    #     matrix[0, 2] = parameters[1]
    #     matrix[1, 2] = -1*parameters[0]

    #     matrix[0, 3] = parameters[3]
    #     matrix[1, 3] = parameters[4]
    #     matrix[2, 3] = -1*parameters[5]
    # else:
    #     print(f"Unexpected number of parameters in {file_path}")

    return matrix

def process_transforms(input_folder, output_folder):
    all_data = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.tfm'):
            file_path = os.path.join(input_folder, file_name)
            matrix = read_transform_file(file_path)
            for i in range(4):
                row = [file_name if i == 0 else ""] + matrix[i, :].tolist()
                all_data.append(row)
    
    # Create a DataFrame with appropriate column names
    columns = ['file_name', 'm_0', 'm_1', 'm_2', 'm_3']
    df = pd.DataFrame(all_data, columns=columns)
    
    output_file = os.path.join(output_folder, 'transforms.csv')
    df.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description="Process .tfm files and save to CSV.")
    parser.add_argument('--input_folder', type=str, required=True, help="Folder containing .tfm files")
    parser.add_argument('--output_folder', type=str, required=True, help="Folder to save the CSV file")
    
    args = parser.parse_args()
    process_transforms(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
