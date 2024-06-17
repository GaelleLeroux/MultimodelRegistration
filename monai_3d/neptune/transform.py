from monai.transforms import (  
    LoadImaged,
    Compose,
    RandCropByPosNegLabeld,
    EnsureChannelFirstd,
    RandRotate90d,
    ScaleIntensityd,
    Resized,
    ToTensord,
    CropForegroundd,
    Orientationd
)   



class TrainTransforms:
    def __init__(self,x:int=32,y:int=64,z:int=64):
        self.train_transform = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"]),
                ScaleIntensityd(keys=["img", "seg"]),
                CropForegroundd(keys=["img", "seg"], source_key="img"),
                Orientationd(keys=["img", "seg"], axcodes="RAS"),
                Resized(keys=["img", "seg"], spatial_size=(x, y, z)),  # Ajustez la taille selon vos besoins
                RandCropByPosNegLabeld(
                    keys=["img", "seg"],
                    label_key="seg",
                    spatial_size=(x/2, y/2, z/2),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="img",
                    image_threshold=0,
                ),
                ToTensord(keys=["img", "seg"]),
            ]
        )

    def __call__(self, inp):
        try:
            y = self.train_transform(inp)
            return y
        except Exception as e:
            print("Erreur lors de l'application de la transformation:", e)
            print("Dictionnaire d'entrée :", inp)
            raise e
        
class ValidTransforms:
    def __init__(self,x:int=32,y:int=64,z:int=64):
        self.valid_transform = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"]),
                ScaleIntensityd(keys=["img", "seg"]),
                CropForegroundd(keys=["img", "seg"], source_key="img"),
                Orientationd(keys=["img", "seg"], axcodes="RAS"),
                Resized(keys=["img", "seg"], spatial_size=(x, y, z)),  # Ajustez la taille selon vos besoins
                ToTensord(keys=["img", "seg"]),
            ]
        )

    def __call__(self, inp):
        try:
            y = self.valid_transform(inp)
            return y
        
        except Exception as e:
            print("Erreur lors de l'application de la transformation:", e)
            print("Dictionnaire d'entrée :", inp)
            raise e



