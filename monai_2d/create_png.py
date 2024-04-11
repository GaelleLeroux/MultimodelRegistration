import nibabel as nib
import os
from PIL import Image
import argparse

def create_png(nii_gz_file_path,output_dir,filename):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Charger les données du fichier NIfTI
    nii_data = nib.load(nii_gz_file_path)

    # Extraire les données d'image sous forme d'un tableau numpy
    image_data = nii_data.get_fdata()

    # Itérer sur la troisième dimension (z) pour sauvegarder chaque tranche
    for z in range(image_data.shape[2]):
        # Extraire la tranche z
        slice_z = image_data[:, :, z]
        
        # Normaliser les données pour qu'elles soient dans l'intervalle [0, 255]
        slice_normalized = ((slice_z - slice_z.min()) / (slice_z.max() - slice_z.min()) * 255).astype('uint8')
        
        # Convertir en image PIL et sauvegarder
        img = Image.fromarray(slice_normalized)
        img.save(os.path.join(output_dir, f'{filename}_slice_{z:03}.png'))

    print(f'Toutes les tranches ont été sauvegardées dans {output_dir}')
    
def main(args):
    input_folder = args.input
    output_general = args.output
    # Get all nifti files in the folder
    nifti_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nifti_files.append(os.path.join(root, file))

    # Get nifti info for every nifti file
    for file in nifti_files:
        base_name = os.path.basename(file)  
        file_name_without_extension = os.path.splitext(os.path.splitext(base_name)[0])[0]
        output_folder = os.path.join(output_general,file_name_without_extension)
        create_png(file,output_folder,file_name_without_extension)
    print("All the files has been treated")
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--input', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/2D_Training/z06_center/label/', help='Input folder')
    parser.add_argument('--output', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/2D_Training/LabelsTr/', help='Output directory tosave the png')
    args = parser.parse_args()



    main(args)