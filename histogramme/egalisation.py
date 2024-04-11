import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt  # Ajout pour la visualisation
import os
import argparse

def CorrectHisto(filepath, outpath,  path_histo = ".",min_porcent=0.01, max_porcent=0.95, i_min=-1500, i_max=4000):
    print("Correcting scan contrast:", filepath)
    input_img = sitk.ReadImage(filepath)
    input_img = sitk.Cast(input_img, sitk.sitkFloat32)
    img = sitk.GetArrayFromImage(input_img)

    # Stocker l'image originale pour comparaison
    original_img = np.copy(img)

    img_min = np.min(img)
    img_max = np.max(img)
    img_range = img_max - img_min

    definition = 1000
    histo = np.histogram(img, bins=definition)
    cum = np.cumsum(histo[0])
    cum = cum - np.min(cum)
    cum = cum / np.max(cum)

    res_high = list(map(lambda i: i > max_porcent, cum)).index(True)
    res_max = (res_high * img_range) / definition + img_min

    res_low = list(map(lambda i: i > min_porcent, cum)).index(True)
    res_min = (res_low * img_range) / definition + img_min

    res_min = max(res_min, i_min)
    res_max = min(res_max, i_max)
    
    


    img = np.where(img > res_max, res_max, img)
    img = np.where(img < res_min, res_min, img)

    output = sitk.GetImageFromArray(img)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)
    
    img_corrected = sitk.GetArrayFromImage(output)

    # Afficher les images avant et après correction côte à côte
    plt.figure(figsize=(10, 5))
    plt.hist(img.flatten(), bins=definition, color='blue', alpha=0.5, label='Avant correction')
    plt.hist(img_corrected.flatten(), bins=definition, color='red', alpha=0.5, label='Après correction')
    plt.title('Histogramme des valeurs de pixels')
    plt.xlabel('Valeur de pixel')
    plt.ylabel('Fréquence')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(path_histo)
    plt.close()

    # Sauvegarde de l'image corrigée
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)

    # return output, comparison_path  # Retourne le chemin vers l'image de comparaison


def main(args):
    input_folder = args.input
    # Get all nifti files in the folder
    nifti_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                nifti_files.append(os.path.join(root, file))
                
    for file in nifti_files :
        filename = os.path.splitext(os.path.basename(file))[0]
        filename = os.path.splitext(filename)[0]
        # print("filename : ",filename)
        # print("file : ",file)
        outpath_png = os.path.join(args.output_png ,filename)+".png"
        output_data = os.path.join(args.output_data,os.path.basename(file))
        # print(f"output_data : {output_data}")
        # print(f"path_histo : {os.path.join(outpath_png,filename)}.png")
        CorrectHisto(file,output_data,outpath_png)
        
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--input', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/imagesTr/', help='Input folder')
    parser.add_argument('--output_png', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/z05_correct_histo_image/comparaison/', help='Output directory for savaing comparaison histo')
    parser.add_argument('--output_data', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/Training/z05_correct_histo_image/data/', help='Output directory for saving data')
    args = parser.parse_args()

    # if output folder does not exist, create it
    if not os.path.exists(args.output_png):
        os.makedirs(args.output_png)
        
    if not os.path.exists(args.output_data):
        os.makedirs(args.output_data)

    main(args)
        