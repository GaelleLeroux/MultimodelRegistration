import os
import argparse
import SimpleITK as sitk
import pandas as pd
import numpy as np

def read_transform_file(file_path):
    transform = sitk.ReadTransform(file_path)
    parameters = [round(p, 2) for p in transform.GetParameters()]
    fixed_parameters = [round(fp, 2) for fp in transform.GetFixedParameters()]
    
    # Initialize a 4x4 identity matrix
    matrix = np.eye(4)
    
    if len(parameters) == 12:
        # 3D transformation matrix
        matrix[:3, :3] = np.array(parameters[:9]).reshape((3, 3))
        matrix[:3, 3] = np.array(parameters[9:12])
    elif len(parameters) == 6:
        # 2D transformation matrix (extend to 4x4)
        matrix[:2, :3] = np.array(parameters).reshape((2, 3))
        matrix[2, 2] = 1
        matrix[3, 3] = 1
    else:
        print(f"Unexpected number of parameters in {file_path}")

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
