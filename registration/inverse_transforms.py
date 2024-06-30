import os
import argparse
import SimpleITK as sitk

def invert_transformations(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.tfm'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read the transformation
            transform = sitk.ReadTransform(input_path)

            # Invert the transformation
            inverted_transform = transform.GetInverse()

            # Write the inverted transformation
            sitk.WriteTransform(inverted_transform, output_path)
            print(f"Inverted {filename} and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Invert .tfm files from input folder and save them to output folder.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing .tfm files.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder to save inverted .tfm files.')

    args = parser.parse_args()

    invert_transformations(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
