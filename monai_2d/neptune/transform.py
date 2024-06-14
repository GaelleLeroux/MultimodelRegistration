from monai.transforms import (  
    LoadImaged,
    Compose,
    RandCropByPosNegLabeld,
    EnsureChannelFirstd,
    RandRotate90d,
    ScaleIntensityd,
    Resized,
    ToTensord
)   



class TrainTransforms:
    def __init__(self, height: int = 256, size=128, pad=16):
        self.train_transform = Compose(
            [
                # LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"],channel_dim=1,strict_check=True),
                ScaleIntensityd(keys=["img", "seg"]),
                RandCropByPosNegLabeld(
                    keys=["img", "seg"], label_key="seg", spatial_size=[300, 300], pos=1, neg=1, num_samples=4
                ),
                RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
                ToTensord(keys=["img", "seg"])
                
            ]
        )

    def __call__(self, inp):
        # print(f"Before transform: {inp}")  # Ajoutez cette ligne pour déboguer les données d'entrée
        # try:
        y = self.train_transform(inp)
        # print(f"After transform: {y[0]["img"].shape}")  # Ajoutez cette ligne pour vérifier les données après transformation
        # print(f"\033[94m Output dictionary after Traintransform: {[(k, v.shape) for k, v in y.items()]} \033[0m")  # Debug output data shapes
        return y
        # except Exception as e:
        #     print("Erreur lors de l'application de la transformation:", e)
        #     print("Dictionnaire d'entrée :", inp)
        #     raise e
        
class ValidTransforms:
    def __init__(self, height: int = 256, size=128, pad=16):
        self.valid_transform = Compose(
            [
                # LoadImaged(keys=["img", "seg"]),
                
                EnsureChannelFirstd(keys=["img", "seg"],channel_dim=1,strict_check=True),
                ScaleIntensityd(keys=["img", "seg"]),
                Resized(keys=["img", "seg"], spatial_size=(300, 300)),
                ToTensord(keys=["img", "seg"])
            ]
        )

    def __call__(self, inp):
        # print(f"Before transform: {inp}")  # Ajoutez cette ligne pour déboguer les données d'entrée
        try:
            y = self.valid_transform(inp)
            # print(f"After transform: {y[0]["img"].shape}")  # Ajoutez cette ligne pour vérifier les données après transformation
            # print(f"\033[94m Output dictionary after Validtransform: {[(k, v.shape) for k, v in y.items()]} \033[0m")
            return y
        except Exception as e:
            print("Erreur lors de l'application de la transformation:", e)
            print("Dictionnaire d'entrée :", inp)
            raise e



