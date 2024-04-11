from PIL import Image
import os
import csv
import argparse
import numpy as np

def create_csv(args):

    root_dir = args.input
    csv_file_path = args.output
    data_for_csv = []

    
    for patient_folder_name in os.listdir(root_dir):
        patient_folder_path = os.path.join(root_dir, patient_folder_name)
        
        if os.path.isdir(patient_folder_path):
            patient_data = [patient_folder_name]
            
            for image_file in os.listdir(patient_folder_path):
                image_path = os.path.join(patient_folder_path, image_file)
                
                if image_file.endswith('.png'):
                    with Image.open(image_path) as img:
                        if args.size == "1":
                            width, height = img.size
                            patient_data.append(f"{width}x{height}")
                        else :
                            amin = np.amin(img)
                            amax = np.amax(img)
                            patient_data.append(f"{amin} / {amax}")
                              
            # Ajouter la ligne complète pour ce patient aux données pour le CSV
            data_for_csv.append(patient_data)

    # Trouver le nombre maximum d'images parmi tous les patients pour déterminer le nombre de colonnes
    max_images = max(len(patient_data) for patient_data in data_for_csv)

    # Écrire les données dans le fichier CSV
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Créer et écrire l'en-tête avec des colonnes dynamiquement nommées selon le nombre d'images
        if args.size == "1":
            header = ['Nom du patient'] + [f'Image {i+1} Dimensions' for i in range(max_images-1)]
        else : 
            header = ['Nom du patient'] + [f'Image {i+1} Min/Max' for i in range(max_images-1)]
        writer.writerow(header)
        
        # Écrire les données de chaque patient
        for patient_data in data_for_csv:
            writer.writerow(patient_data)

    print(f'Le fichier CSV a été créé avec succès : {csv_file_path}')
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--input', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/2D_Training/Image/', help='Input folder')
    parser.add_argument('--output', type=str, default='/home/luciacev/Documents/Gaelle/Data/MultimodelReg/2D_Training/info_min_max_image.csv', help='Output directory tosave the png')
    parser.add_argument('--size', type=str, default='1', help='1 if saving size in csv or 0 if saving min,max')
    args = parser.parse_args()



    create_csv(args)
