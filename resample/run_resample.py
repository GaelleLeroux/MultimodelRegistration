import subprocess
import csv
import argparse

import csv
# THE PACING IS CHOOSEN RUNING CALCUL_SPACING_MEAN.PY DONC LA MOYENNE DES SPACING QUE ON A

def main(args):
    # Remplacez ceci par le chemin vers votre fichier CSV
    csv_file_path = args.csv
    # Ouvrir le fichier CSV en lecture
    #RESAMPLE
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
            # command = [f"python3 {args.python_file} --img \"{input_path}\" --out \"{out_path}\" --size 768 576 768 --spacing 0.3 0.3 0.3 --center False --linear False --fit_spacing True --image_dimension 3 --pixel_dimension 1 --rgb False --ow 0"]
            # command = [f"python3 {args.python_file} --img \"{input_path}\" --out \"{out_path}\" --size 443 443 119 --spacing 0.3 0.3 0.3 --center False --linear False --fit_spacing True --image_dimension 3 --pixel_dimension 1 --rgb False --ow 0"]
            command = [f"python3 {args.python_file} --img \"{input_path}\" --out \"{out_path}\" --size 119 443 443  --fit_spacing True --center False --linear False --image_dimension 3 --pixel_dimension 1 --rgb False --ow 0"]
            subprocess.run(command,shell=True)
            


if __name__=="__main__":
    # SIZE AND SPACING TO RESAMPLE ARE HARD WRITTEN IN THE LINE 24
    parser = argparse.ArgumentParser(description='Get nifti info')
    parser.add_argument('--csv', required=True,type=str, help='name of the CSV')
    parser.add_argument('--python_file', default="/home/luciacev/Documents/Gaelle/MultimodelRegistration/resample/resample.py",type=str, help='path to resample.py')
    # /home/luciacev/Documents/Gaelle/MultimodelRegistration/resample/resample.py
    args = parser.parse_args()


    main(args)