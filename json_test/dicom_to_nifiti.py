# import os

# import SimpleITK as sitk
# import dicom2nifti
# import glob

# def search(path,*args):
#     """
#     Return a dictionary with args element as key and a list of file in path directory finishing by args extension for each key

#     Example:
#     args = ('json',['.nii.gz','.nrrd'])
#     return:
#         {
#             'json' : ['path/a.json', 'path/b.json','path/c.json'],
#             '.nii.gz' : ['path/a.nii.gz', 'path/b.nii.gz']
#             '.nrrd.gz' : ['path/c.nrrd']
#         }
#     """
#     arguments=[]
#     for arg in args:
#         if type(arg) == list:
#             arguments.extend(arg)
#         else:
#             arguments.append(arg)
#     return {key: [i for i in glob.iglob(os.path.normpath("/".join([path,'**','*'])),recursive=True) if i.endswith(key)] for key in arguments}

# def convertdicom2nifti(input_folder,output_folder=None):
#     patients_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder,folder)) and folder != 'NIFTI']

#     if output_folder is None:
#         output_folder = os.path.join(input_folder,'NIFTI')

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     for patient in patients_folders:
#         if not os.path.exists(os.path.join(output_folder,patient+".nii.gz")):
#             print("Converting patient: {}...".format(patient))
#             current_directory = os.path.join(input_folder,patient)
#             print("current directory : ",current_directory)
#             try:
#                 reader = sitk.ImageSeriesReader()
#                 sitk.ProcessObject_SetGlobalWarningDisplay(False)
#                 dicom_names = reader.GetGDCMSeriesFileNames(current_directory)
#                 reader.SetFileNames(dicom_names)
#                 image = reader.Execute()
#                 sitk.ProcessObject_SetGlobalWarningDisplay(True)
#                 sitk.WriteImage(image, os.path.join(output_folder,os.path.basename(current_directory)+'.nii.gz'))
#             except RuntimeError:
#                 dicom2nifti.convert_directory(current_directory,output_folder)
#                 nifti_file = search(output_folder,'nii.gz')['nii.gz'][0]
#                 os.rename(nifti_file,os.path.join(output_folder,patient+".nii.gz"))

# print("Conversion terminée.")
# input_folder = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/test'
# output_folder = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/test2'
# convertdicom2nifti(input_folder, output_folder)


import os
import SimpleITK as sitk

def convert_dicom_to_nifti(input_folder, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'NIFTI')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Liste tous les dossiers patients dans le dossier d'entrée
    patients_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]

    for patient in patients_folders:
        output_file_path = os.path.join(output_folder, patient + ".nii.gz")
        if not os.path.exists(output_file_path):
            print(f"Converting patient: {patient}...")
            current_directory = os.path.join(input_folder, patient)

            try:
                # Lecture des séries DICOM
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(current_directory)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()

                # Écriture de l'image en format NIfTI
                sitk.WriteImage(image, output_file_path)
                print(f"Conversion réussie pour: {patient}")
            except RuntimeError as e:
                print(f"Erreur lors de la conversion pour {patient}: {e}")

print("Conversion terminée.")
input_folder = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/test'
output_folder = '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/test2'
convert_dicom_to_nifti(input_folder, output_folder)
