import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process transformation matrices.')
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing .tfm files')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the output PNG file')
    parser.add_argument('--name_png', type=str, default="comput_matrix", help='Name of the output PNG file')
    return parser.parse_args()

def read_tfm_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('Parameters:'):
                parameters = list(map(float, line.split()[1:]))
                # Apply LPS to RAS transformation
                parameters[0] = -parameters[0]  # Flip the x-coordinate
                parameters[1] = -parameters[1]  # Flip the y-coordinate
                print("files : ",file_path)
                print("parameters : ",parameters)
                return parameters
    return None

def calculate_statistics(parameters_list):
    parameters_array = np.array(parameters_list)
    avg_params = np.mean(parameters_array, axis=0)
    sd_params = np.std(parameters_array, axis=0)
    range_params = np.ptp(parameters_array, axis=0)
    
    return avg_params, sd_params, range_params

def save_statistics_as_image(output_folder, name_png, stats):
    avg_params, sd_params, range_params = stats
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    stats_text = f"""
    Parameters (RAS):
    Average: {avg_params}
    SD: {sd_params}
    Range: {range_params}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=12, va='center')
    output_path = os.path.join(output_folder, name_png)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    args = parse_arguments()
    input_folder = args.input_folder
    output_folder = args.output_folder
    name_png = args.name_png

    tfm_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tfm')]
    if not tfm_files:
        print("No .tfm files found in the input folder.")
        return

    parameters_list = [read_tfm_file(file) for file in tfm_files]
    parameters_list = [params for params in parameters_list if params is not None]
    if not parameters_list:
        print("No valid transformation parameters found.")
        return

    stats = calculate_statistics(parameters_list)
    save_statistics_as_image(output_folder, name_png, stats)

if __name__ == "__main__":
    main()
