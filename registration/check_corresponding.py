import os
import argparse

def get_corresponding_file(folder, patient_id, modality):
    """Get the corresponding file for a given patient ID and modality."""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith(patient_id) and modality in file and file.endswith(".nii.gz"):
                return os.path.join(root, file)
    return None

def main(input_folder_1, input_folder_2):
    for root, _, files in os.walk(input_folder_1):
        for file in files:
                patient_id = file.split("_MR_")[0]
                cbct_path_original = get_corresponding_file(input_folder_2, patient_id, "_MR_")
                
                if cbct_path_original:
                    print(f"Found corresponding file for patient {patient_id}: {cbct_path_original}")
                    # pass
                else:
                    print(f"No corresponding file found for patient {patient_id}") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find corresponding files in input folders.")
    parser.add_argument("--input_folder_1", type=str, help="The path to the first input folder.")
    parser.add_argument("--input_folder_2", type=str, help="The path to the second input folder.")
    
    args = parser.parse_args()
    
    main(args.input_folder_1, args.input_folder_2)
