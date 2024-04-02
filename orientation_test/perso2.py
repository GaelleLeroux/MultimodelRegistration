import nibabel as nib


input_file = "/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/center/M004_MRI.nii.gz"
    
nifti_img = nib.load(input_file)

data = nifti_img.get_fdata()

y_slice = data[:, 1, :]

print("y_slice : ",y_slice)

z_slice = data[:, :, 1]