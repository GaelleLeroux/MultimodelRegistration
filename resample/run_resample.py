import subprocess
import csv


import csv
# THE PACING IS CHOOSEN RUNING CALCUL_SPACING_MEAN.PY DONC LA MOYENNE DES SPACING QUE ON A


# Remplacez ceci par le chemin vers votre fichier CSV
csv_file_path = '/home/luciacev/Documents/Gaelle/MultimodelRegistration/resample/resample_spacing_label.csv'

# Ouvrir le fichier CSV en lecture
with open(csv_file_path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    # Boucle sur chaque ligne du fichier CSV
    for row in csv_reader:
        # Afficher les informations de chaque ligne
        size = tuple(map(int, row["size"].strip("()").split(",")))
        input_path = row["in"]
        out_path = row["out"]
        print(size[1])
        print(f'Image d\'entrée: {row["in"]}, Image de sortie: {row["out"]}, Taille: {size}')
        command = [f"python3 /home/luciacev/Documents/Gaelle/MultimodelRegistration/resample/resample.py --img \"{input_path}\" --out \"{out_path}\" --size 224 224 {str(size[2])} --spacing 0.46 0.46 2.86 --linear False --fit_spacing True --image_dimension 3 --pixel_dimension 1 --rgb False --ow 0"]
        subprocess.run(command,shell=True)
