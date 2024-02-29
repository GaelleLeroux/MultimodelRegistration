import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

def convert_nifti_to_dicom(nifti_file_path, output_folder, patient_name, patient_id, study_uid=None):
    # Charger le fichier NIfTI
    nifti_img = nib.load(nifti_file_path)
    data = nifti_img.get_fdata()

    header = nifti_img.header
    pixdim = header['pixdim'][1:3]

    # Générer un StudyInstanceUID unique pour l'ensemble de l'étude, si non fourni
    if study_uid is None:
        study_uid = generate_uid()

    # Supposons que data est une image 3D: [Height, Width, Depth]
    for i in range(data.shape[2]):
        # Créer un Dataset DICOM de base
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # Ajouter des métadonnées essentielles
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.StudyInstanceUID = study_uid

        # Ajouter les données d'image (coupe i)
        slice_data = data[:, :, i].astype(np.uint16)
        ds.PixelData = slice_data.tobytes()

        # Définir les attributs requis par DICOM pour les images
        ds.Rows = slice_data.shape[0]
        ds.Columns = slice_data.shape[1]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.Modality = "MR"
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.HighBit = 15
        ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        ds.SOPInstanceUID = generate_uid()

        ds.PixelSpacing = [str(pixdim[0]), str(pixdim[1])]
        ds.SliceThickness = str(pixdim[2])

        # Enregistrer le fichier DICOM
        dicom_file_path = f"{output_folder}/slice_{i+1:04d}"
        ds.save_as(dicom_file_path, write_like_original=False)
        print(f"Saved DICOM slice {i+1}")

# Exemple d'utilisation
convert_nifti_to_dicom('/home/luciacev/Documents/Gaelle/Data/MultimodelReg/nfiti_to_dicom/original/M002_original.nii.gz', '/home/luciacev/Documents/Gaelle/Data/MultimodelReg/nfiti_to_dicom/output2','ABC', '123456')
