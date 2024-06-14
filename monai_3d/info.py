import nibabel as nib
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
# Replace 'your_file_path.nii.gz' with the path to your NIfTI file
file_path = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/imagesTr/M004_MRI.nii.gz'


data_dir = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/"

train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))

# Load the NIfTI file
nifti_image = nib.load(file_path)

# Get the header from the loaded NIfTI file
header = nifti_image.header

# Read spacing from the header (voxel sizes)
spacing = header.get_zooms()

# Orientation information is a bit more complex. NIfTI images are usually stored
# in RAS+ orientation, but the exact orientation can be inferred from the affine matrix.
affine = nifti_image.affine

# Extraction des données de l'image
image_data = nifti_image.get_fdata()
print("Image size:", image_data.shape)

# Print the results
# print("Voxel Spacing (in mm):", spacing)
# print("Affine Matrix (orientation and position):")
# print(affine)



###########################################################################################################################################################################333
'''
EXECL SIZE
'''

# Directory containing NIfTI files
data_dir = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/"
data_dir = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/example_monai/Task09_Spleen/"
nifti_files = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))

# Initialize a list to store file data
file_data = []

# Traverse through all NIfTI files
for file_path in nifti_files:
    # Load the NIfTI file
    nifti_image = nib.load(file_path)
    # Get the image data
    image_data = nifti_image.get_fdata()
    
    # Extract file name from the file path
    file_name = os.path.basename(file_path)
    # Extract image size
    image_size = image_data.shape
    
    # Append file name and image size to the list
    file_data.append([file_name] + list(image_size))

# Create a DataFrame
df = pd.DataFrame(file_data, columns=['FileName', 'DimX', 'DimY', 'DimZ'])

# Optionally, if your images could be 4D, adjust the columns as follows:
# df = pd.DataFrame(file_data, columns=['FileName', 'DimX', 'DimY', 'DimZ', 'DimT']) # Adjust columns for 4D

# Save the DataFrame to an Excel file
excel_path = os.path.join(data_dir, "nifti_file_sizes.xlsx")
df.to_excel(excel_path, index=False)

print(f"Excel file created at {excel_path}")


###########################################################################################################################################################################333
'''
AMIN et AMAX
'''
# # Calcul des valeurs minimales et maximales
# a_min = image_data.min()
# a_max = image_data.max()

# print(f'a_min: {a_min}, a_max: {a_max}')


# print("*"*150)

# data_dir = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/"
# # data_dir = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/example_monai/Task09_Spleen/"
# train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))

# # Prepare lists to store min and max values and image names
# amin_values = []
# amax_values = []
# image_names = []

# # Loop through all image paths
# for image_path in train_images:
#     # Load the NIfTI file
#     nifti_image = nib.load(image_path)
    
#     # Extract image data
#     image_data = nifti_image.get_fdata()
    
#     # Calculate min and max
#     a_min = image_data.min()
#     a_max = image_data.max()
    
#     # Store the values and the image name
#     amin_values.append(a_min)
#     amax_values.append(a_max)
#     image_names.append(os.path.basename(image_path))
    
    
# print("moyenne a_min : ",np.mean(amin_values))
# print("moyenne a_max : ",np.mean(amax_values))

# # Plotting
# fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# # Min values plot
# axs[0].bar(image_names, amin_values, color='skyblue')
# axs[0].set_title('Minimum Values per Image')
# axs[0].set_ylabel('a_min')
# axs[0].tick_params(axis='x', rotation=90)

# # Max values plot
# axs[1].bar(image_names, amax_values, color='salmon')
# axs[1].set_title('Maximum Values per Image')
# axs[1].set_ylabel('a_max')
# axs[1].tick_params(axis='x', rotation=90)

# plt.tight_layout()
# plt.show()

