import os
import argparse
import SimpleITK as sitk
import pandas as pd
import math

def radians_to_degrees(radians):
    return radians * (180 / math.pi)

def read_transform_file(file_path):
    transform = sitk.ReadTransform(file_path)
    parameters = [p for p in transform.GetParameters()]
    fixed_parameters = [fp for fp in transform.GetFixedParameters()]
    
    return parameters, fixed_parameters

def process_transforms(input_folder, output_folder,csv_name):
    data = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.tfm'):
            file_path = os.path.join(input_folder, file_name)
            parameters, fixed_parameters = read_transform_file(file_path)
            file_name = file_name.split('_MR_')[0]
            transform_data = [file_name] + [radians_to_degrees(parameters[0])] +[radians_to_degrees(parameters[1])] + [radians_to_degrees(parameters[2])] +parameters[3:] 
            data.append(transform_data)
    
    # Create a DataFrame with appropriate column names
    max_len = max(len(row) for row in data)
    columns = ['Case'] + ['Rotation_X']+ ['Rotation_Y'] + ['Rotation_Z'] + ['Translation_X'] +['Translation_Y'] + ['Translation_Z']
    df = pd.DataFrame(data, columns=columns)
    
    output_file = os.path.join(output_folder, f'{csv_name}.csv')
    df.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description="Process .tfm files and save to CSV.")
    parser.add_argument('--input_folder', type=str, help="Folder containing .tfm files")
    parser.add_argument('--output_folder', type=str, help="Folder to save the CSV file")
    parser.add_argument('--name_csv', type=str, help="Name of the CSV file")
    
    args = parser.parse_args()
    process_transforms(args.input_folder, args.output_folder, args.name_csv)

if __name__ == "__main__":
    main()
