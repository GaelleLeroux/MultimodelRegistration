import os
import pydicom
import json
import base64

def dicom_to_json(source_directory, target_directory):
    """
    Lit tous les fichiers DICOM d'un répertoire donné, les convertit en JSON,
    et sauvegarde les fichiers JSON dans un autre répertoire.

    Args:
    - source_directory (str): Chemin vers le répertoire source contenant les fichiers DICOM.
    - target_directory (str): Chemin vers le répertoire cible pour sauvegarder les fichiers JSON.
    """
    # Vérifier si le répertoire cible existe, sinon le créer
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Parcourir tous les fichiers du répertoire source
    for filename in os.listdir(source_directory):
        if filename.lower().endswith('.dcm'):  # Filtrer pour ne traiter que les fichiers DICOM
            dicom_path = os.path.join(source_directory, filename)
            json_path = os.path.join(target_directory, filename.replace('.dcm', '.json'))

            # Charger le fichier DICOM
            dicom_file = pydicom.dcmread(dicom_path)

            # Extraire les métadonnées sous forme de dictionnaire, en excluant les valeurs binaires
            dicom_metadata = {element.keyword: element.value for element in dicom_file if element.keyword and not isinstance(element.value, pydicom.multival.MultiValue)}

            # Convertir le dictionnaire en chaîne JSON et l'enregistrer dans un fichier
            with open(json_path, 'w') as json_file:
                json.dump(dicom_metadata, json_file, indent=4)

    print(f"Tous les fichiers DICOM de '{source_directory}' ont été convertis en JSON et sauvegardés dans '{target_directory}'.")

# Exemple d'utilisation (les chemins sont à adapter selon les besoins)
# dicom_to_json('/chemin/vers/repertoire/source', '/chemin/vers/repertoire/cible')

def dicom_to_json_v2(source_directory, target_directory):
    """
    Lit tous les fichiers d'un répertoire donné, vérifie s'ils sont des fichiers DICOM (indépendamment de leur extension),
    les convertit en JSON, et sauvegarde les fichiers JSON dans un autre répertoire.

    Args:
    - source_directory (str): Chemin vers le répertoire source contenant les fichiers potentiellement DICOM.
    - target_directory (str): Chemin vers le répertoire cible pour sauvegarder les fichiers JSON.
    """
    # Vérifier si le répertoire cible existe, sinon le créer
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Parcourir tous les fichiers du répertoire source
    for filename in os.listdir(source_directory):
        full_path = os.path.join(source_directory, filename)
        try:
            # Tenter de lire le fichier comme un fichier DICOM
            dicom_file = pydicom.dcmread(full_path, force=True)  # Utiliser force=True pour lire sans tenir compte de l'extension
            # Extraire les métadonnées sous forme de dictionnaire, en excluant les valeurs binaires
            dicom_metadata = {element.keyword: element.value for element in dicom_file if element.keyword and not isinstance(element.value, pydicom.multival.MultiValue)}

            # Générer un nom de fichier JSON en ajoutant l'extension .json
            json_filename = f"{filename}.json"
            json_path = os.path.join(target_directory, json_filename)

            # Convertir le dictionnaire en chaîne JSON et l'enregistrer dans un fichier
            with open(json_path, 'w') as json_file:
                json.dump(dicom_metadata, json_file, indent=4)
        except Exception as e:
            # Si le fichier n'est pas un DICOM valide, passer au fichier suivant
            print(f"Erreur lors du traitement du fichier {filename}: {e}")
            continue

    print(f"Tous les fichiers DICOM de '{source_directory}' ont été tentés d'être convertis en JSON et sauvegardés dans '{target_directory}', si possible.")

def dicom_to_json_v3(source_directory, target_directory):
    """
    Lit tous les fichiers d'un répertoire donné, vérifie s'ils sont des fichiers DICOM (indépendamment de leur extension),
    les convertit en JSON, et sauvegarde les fichiers JSON dans un autre répertoire.

    Args:
    - source_directory (str): Chemin vers le répertoire source contenant les fichiers potentiellement DICOM.
    - target_directory (str): Chemin vers le répertoire cible pour sauvegarder les fichiers JSON.
    """
    # Vérifier si le répertoire cible existe, sinon le créer
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Parcourir tous les fichiers du répertoire source
    for filename in os.listdir(source_directory):
        full_path = os.path.join(source_directory, filename)
        try:
            # Tenter de lire le fichier comme un fichier DICOM
            dicom_file = pydicom.dcmread(full_path, force=True)
            dicom_metadata = {}
            for element in dicom_file:
                if element.keyword:
                    # Convertir les types spéciaux (comme PersonName) en chaînes
                    if isinstance(element.value, pydicom.valuerep.PersonName):
                        dicom_metadata[element.keyword] = str(element.value)
                    elif not isinstance(element.value, pydicom.multival.MultiValue):
                        dicom_metadata[element.keyword] = element.value
            # Générer un nom de fichier JSON en ajoutant l'extension .json
            json_filename = f"{filename}.json"
            json_path = os.path.join(target_directory, json_filename)

            # Convertir le dictionnaire en chaîne JSON et l'enregistrer dans un fichier
            with open(json_path, 'w') as json_file:
                json.dump(dicom_metadata, json_file, indent=4)
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {filename}: {e}")
            continue

    print(f"Tous les fichiers DICOM de '{source_directory}' ont été tentés d'être convertis en JSON et sauvegardés dans '{target_directory}', si possible.")

def dicom_to_json_v4(source_directory, target_directory):
    """
    Lit tous les fichiers d'un répertoire donné, vérifie s'ils sont des fichiers DICOM (indépendamment de leur extension),
    les convertit en JSON, et sauvegarde les fichiers JSON dans un autre répertoire.

    Args:
    - source_directory (str): Chemin vers le répertoire source contenant les fichiers potentiellement DICOM.
    - target_directory (str): Chemin vers le répertoire cible pour sauvegarder les fichiers JSON.
    """
    # Vérifier si le répertoire cible existe, sinon le créer
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Parcourir tous les fichiers du répertoire source
    for filename in os.listdir(source_directory):
        full_path = os.path.join(source_directory, filename)
        try:
            # Tenter de lire le fichier comme un fichier DICOM
            dicom_file = pydicom.dcmread(full_path, force=True)
            dicom_metadata = {}
            for element in dicom_file:
                if element.keyword:
                    # Convertir les types spéciaux en chaînes ou en base64
                    if isinstance(element.value, pydicom.valuerep.PersonName):
                        dicom_metadata[element.keyword] = str(element.value)
                    elif isinstance(element.value, bytes):
                        # Encoder les données binaires en base64
                        dicom_metadata[element.keyword] = base64.b64encode(element.value).decode('utf-8')
                    elif not isinstance(element.value, pydicom.multival.MultiValue):
                        dicom_metadata[element.keyword] = element.value
            # Générer un nom de fichier JSON en ajoutant l'extension .json
            json_filename = f"{filename}.json"
            json_path = os.path.join(target_directory, json_filename)

            # Convertir le dictionnaire en chaîne JSON et l'enregistrer dans un fichier
            with open(json_path, 'w') as json_file:
                json.dump(dicom_metadata, json_file, indent=4)
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {filename}: {e}")
            continue

    print(f"Tous les fichiers DICOM de '{source_directory}' ont été tentés d'être convertis en JSON et sauvegardés dans '{target_directory}', si possible.")


dicom_to_json_v4("/home/luciacev/Documents/Gaelle/Data/MultimodelReg/UNET-TMJ/raw_data/M007","/home/luciacev/Documents/Gaelle/Data/MultimodelReg/UNET-TMJ/result_data")