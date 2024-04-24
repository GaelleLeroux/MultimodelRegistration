import SimpleITK as sitk
import numpy as np
import argparse
import os
import glob
import sys
import csv

def adjust_origin_to_center_image(img):
    # Calculer le centre actuel de l'image
    size = img.GetSize()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()

    # Calculer le centre physique de l'image
    physical_size = np.array(size) * np.array(spacing)
    current_center = np.array(origin) + physical_size / 2

    # Calculer le nouveau centre souhaité (au milieu de chaque dimension)
    new_center = physical_size / 2

    # Calculer le nouvel origine pour centrer l'image
    new_origin = np.array(origin) + (new_center - current_center)

    # Mettre à jour l'origine de l'image
    img.SetOrigin(new_origin.tolist())

    return img

def center_and_pad_image(img, desired_size):
    # Obtenir les informations actuelles de l'image
    size = img.GetSize()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()

    # Calculer le centre physique actuel de l'image
    physical_size = np.array(size) * np.array(spacing)
    current_center = np.array(origin) + physical_size / 2

    # Calculer la taille physique désirée
    desired_physical_size = np.array(desired_size) * np.array(spacing)
    new_center = desired_physical_size / 2

    # Calculer le nouvel origine pour centrer l'image
    new_origin = np.array(origin) + (new_center - current_center)

    # Créer un filtre pour le padding
    pad_filter = sitk.ConstantPadImageFilter()

    # Calculer les valeurs de padding nécessaires
    lower_padding = [max(0, (ds - s) // 2) for ds, s in zip(desired_size, size)]
    upper_padding = [max(0, ds - s - lp) for ds, s, lp in zip(desired_size, size, lower_padding)]

    # Appliquer le padding
    padded_img = pad_filter.Execute(img, lower_padding, upper_padding, 0) # 0 est la valeur de padding
    padded_img.SetOrigin(new_origin.tolist())

    return padded_img


def center_image_fully_sitk(img):
    # Convertir SimpleITK.Image en numpy array
    data = sitk.GetArrayFromImage(img)

    # Obtenir les dimensions
    sz, sy, sx = data.shape

    # Identifier les tranches non-noires dans les trois dimensions
    non_black_x = np.any(data, axis=(0, 1))
    non_black_y = np.any(data, axis=(0, 2))
    non_black_z = np.any(data, axis=(1, 2))

    # Calculer les indices minimaux et maximaux non-noirs pour chaque axe
    xmin, xmax = np.where(non_black_x)[0][[0, -1]] if np.any(non_black_x) else (0, sx)
    ymin, ymax = np.where(non_black_y)[0][[0, -1]] if np.any(non_black_y) else (0, sy)
    zmin, zmax = np.where(non_black_z)[0][[0, -1]] if np.any(non_black_z) else (0, sz)

    # Déterminer les centres des indices non-noirs
    center_x, center_y, center_z = (xmin + xmax) // 2, (ymin + ymax) // 2, (zmin + zmax) // 2

    # Calculer les décalages nécessaires pour centrer l'image
    shift_x, shift_y, shift_z = sx//2 - center_x, sy//2 - center_y, sz//2 - center_z

    # Appliquer les décalages à chaque dimension
    data = np.roll(data, shift_x, axis=2)
    data = np.roll(data, shift_y, axis=1)
    data = np.roll(data, shift_z, axis=0)

    # Convertir le numpy array centré en SimpleITK.Image
    centered_img = sitk.GetImageFromArray(data)
    centered_img.CopyInformation(img)  # Copier les informations de l'image originale

    return centered_img

def resample_fn(img, args):
    output_size = args.size 
    fit_spacing = args.fit_spacing
    iso_spacing = args.iso_spacing
    pixel_dimension = args.pixel_dimension
    center = args.center

    # if(pixel_dimension == 1):
    #     zeroPixel = 0
    # else:
    #     zeroPixel = np.zeros(pixel_dimension)

    if args.linear:
        InterpolatorType = sitk.sitkLinear
    else:
        InterpolatorType = sitk.sitkNearestNeighbor

    

    spacing = img.GetSpacing()  
    size = img.GetSize()

    output_origin = img.GetOrigin()
    input_origin = img.GetOrigin()
    output_size = [si if o_si == -1 else o_si for si, o_si in zip(size, output_size)]

    if(fit_spacing):
        output_spacing = [sp*si/o_si for sp, si, o_si in zip(spacing, size, output_size)]
    else:
        output_spacing = spacing
        

    if(iso_spacing):
        output_spacing_filtered = [sp for si, sp in zip(args.size, output_spacing) if si != -1]
        # print(output_spacing_filtered)
        max_spacing = np.max(output_spacing_filtered)
        output_spacing = [sp if si == -1 else max_spacing for si, sp in zip(args.size, output_spacing)]
        # print(output_spacing)

    
    if(args.spacing is not None):
        output_spacing = args.spacing

    if(args.origin is not None):
        output_origin = args.origin

    if(center):
        output_physical_size = np.array(output_size)*np.array(output_spacing)
        input_physical_size = np.array(size)*np.array(spacing)
        # output_origin = np.array(output_origin) - (output_physical_size - input_physical_size)/2.0
        output_origin = np.array(output_origin) + input_physical_size/2

    print("Input size:", size)
    print("Input spacing:", spacing)
    print("Output size:", output_size)
    print("Output spacing:", output_spacing)
    print("Output origin:", output_origin)
    print("Input Origin : ",input_origin)

    resampleImageFilter = sitk.ResampleImageFilter()
    resampleImageFilter.SetInterpolator(InterpolatorType)   
    resampleImageFilter.SetOutputSpacing(output_spacing)
    resampleImageFilter.SetSize(output_size)
    resampleImageFilter.SetOutputDirection(img.GetDirection())
    resampleImageFilter.SetOutputOrigin(output_origin)
    
    resampled_img = resampleImageFilter.Execute(img)
    # resampleImageFilter.SetDefaultPixelValue(zeroPixel)
    resampled_img = adjust_origin_to_center_image(resampled_img)

    return resampled_img

def Resample(img_filename, args):

    output_size = args.size 
    fit_spacing = args.fit_spacing
    iso_spacing = args.iso_spacing
    img_dimension = args.image_dimension
    pixel_dimension = args.pixel_dimension

    print("Reading:", img_filename) 
    img = sitk.ReadImage(img_filename)

    if(args.img_spacing):
        img.SetSpacing(args.img_spacing)
        
    resampled_img = resample_fn(img,args)      
    desired_size = [int(sz * osz / sp) for sz, sp, osz in zip(img.GetSize(), img.GetSpacing(), resampled_img.GetSpacing())]
    centered_padded_img = center_and_pad_image(resampled_img, desired_size)
    return centered_padded_img

    # return resample_fn(img, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resample an image', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    in_group = parser.add_mutually_exclusive_group(required=True)

    in_group.add_argument('--img', type=str, help='image to resample')
    in_group.add_argument('--dir', type=str, help='Directory with image to resample')
    in_group.add_argument('--csv', type=str, help='CSV file with column img with paths to images to resample')

    csv_group = parser.add_argument_group('CSV extra parameters')
    csv_group.add_argument('--csv_column', type=str, default='image', help='CSV column name (Only used if flag csv is used)')
    csv_group.add_argument('--csv_root_path', type=str, default=None, help='Replaces a root path directory to empty, this is use to recreate a directory structure in the output directory, otherwise, the output name will be the name in the csv (only if csv flag is used)')
    csv_group.add_argument('--csv_use_spc', type=int, default=0, help='Use the spacing information in the csv instead of the image')
    csv_group.add_argument('--csv_column_spcx', type=str, default=None, help='Column name in csv')
    csv_group.add_argument('--csv_column_spcy', type=str, default=None, help='Column name in csv')
    csv_group.add_argument('--csv_column_spcz', type=str, default=None, help='Column name in csv')

    transform_group = parser.add_argument_group('Transform parameters')
    transform_group.add_argument('--ref', type=str, help='Reference image. Use an image as reference for the resampling', default=None)
    transform_group.add_argument('--size', nargs="+", type=int, help='Output size, -1 to leave unchanged', default=None)
    transform_group.add_argument('--img_spacing', nargs="+", type=float, default=None, help='Use this spacing information instead of the one in the image')
    transform_group.add_argument('--spacing', nargs="+", type=float, default=None, help='Output spacing')
    transform_group.add_argument('--origin', nargs="+", type=float, default=None, help='Output origin')
    transform_group.add_argument('--linear', type=bool, help='Use linear interpolation.', default=False)
    transform_group.add_argument('--center', type=bool, help='Center the image in the space', default=False)
    transform_group.add_argument('--fit_spacing', type=bool, help='Fit spacing to output', default=False)
    transform_group.add_argument('--iso_spacing', type=bool, help='Same spacing for resampled output', default=False)

    img_group = parser.add_argument_group('Image parameters')
    img_group.add_argument('--image_dimension', type=int, help='Image dimension', default=2)
    img_group.add_argument('--pixel_dimension', type=int, help='Pixel dimension', default=1)
    img_group.add_argument('--rgb', type=bool, help='Use RGB type pixel', default=False)

    out_group = parser.add_argument_group('Ouput parameters')
    out_group.add_argument('--ow', type=int, help='Overwrite', default=1)
    out_group.add_argument('--out', type=str, help='Output image/directory', default="./out.nrrd")
    out_group.add_argument('--out_ext', type=str, help='Output extension type', default=None)

    args = parser.parse_args()

    filenames = []
    if(args.img):
        fobj = {}
        fobj["img"] = args.img
        fobj["out"] = args.out
        filenames.append(fobj)
    elif(args.dir):
        out_dir = args.out
        normpath = os.path.normpath("/".join([args.dir, '**', '*']))
        for img in glob.iglob(normpath, recursive=True):
            if os.path.isfile(img) and True in [ext in img for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png"]]:
                fobj = {}
                fobj["img"] = img
                fobj["out"] = os.path.normpath(out_dir + "/" + img.replace(args.dir, ''))
                if args.out_ext is not None:
                    out_ext = args.out_ext
                    if out_ext[0] != ".":
                        out_ext = "." + out_ext
                    fobj["out"] = os.path.splitext(fobj["out"])[0] + out_ext
                if not os.path.exists(os.path.dirname(fobj["out"])):
                    os.makedirs(os.path.dirname(fobj["out"]))
                if not os.path.exists(fobj["out"]) or args.ow:
                    filenames.append(fobj)
    elif(args.csv):
        replace_dir_name = args.csv_root_path
        with open(args.csv) as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                fobj = {}
                fobj["img"] = row[args.csv_column]
                fobj["out"] = row[args.csv_column]
                if(replace_dir_name):
                    fobj["out"] = fobj["out"].replace(replace_dir_name, args.out)
                if args.csv_use_spc:
                    img_spacing = []
                    if args.csv_column_spcx:
                        img_spacing.append(row[args.csv_column_spcx])
                    if args.csv_column_spcy:
                        img_spacing.append(row[args.csv_column_spcy])
                    if args.csv_column_spcz:
                        img_spacing.append(row[args.csv_column_spcz])
                    fobj["img_spacing"] = img_spacing

                if "ref" in row:
                    fobj["ref"] = row["ref"]

                if args.out_ext is not None:
                    out_ext = args.out_ext
                    if out_ext[0] != ".":
                        out_ext = "." + out_ext
                    fobj["out"] = os.path.splitext(fobj["out"])[0] + out_ext
                if not os.path.exists(os.path.dirname(fobj["out"])):
                    os.makedirs(os.path.dirname(fobj["out"]))
                if not os.path.exists(fobj["out"]) or args.ow:
                    filenames.append(fobj)
    else:
        raise "Set img or dir to resample!"

    if(args.rgb):
        if(args.pixel_dimension == 3):
            print("Using: RGB type pixel with unsigned char")
        elif(args.pixel_dimension == 4):
            print("Using: RGBA type pixel with unsigned char")
        else:
            print("WARNING: Pixel size not supported!")

    if args.ref is not None:
        print(args.ref)
        ref = sitk.ReadImage(args.ref)
        args.size = ref.GetSize()
        args.spacing = ref.GetSpacing()
        args.origin = ref.GetOrigin()
        
        print("*"*50)
        print("args.ref : ")
        print(args.ref)

    for fobj in filenames:

        try:

            if "ref" in fobj and fobj["ref"] is not None:
                ref = sitk.ReadImage(fobj["ref"])
                args.size = ref.GetSize()
                args.spacing = ref.GetSpacing()
                args.origin = ref.GetOrigin()
            print("*"*50)
            print("args : ")
            print(args)

            if args.size is not None:
                img = Resample(fobj["img"], args)
            else:
                img = sitk.ReadImage(fobj["img"])

            print("Writing:", fobj["out"])
            writer = sitk.ImageFileWriter()
            writer.SetFileName(fobj["out"])
            writer.UseCompressionOn()
            writer.Execute(img)
            
        except Exception as e:
            print(e, file=sys.stderr)
