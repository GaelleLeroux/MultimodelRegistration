import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import Callback

class SegmentationLogger(Callback):
    def __init__(self, num_images=8, log_steps=30, slice_axis=2):
        self.log_steps = log_steps
        self.num_images = num_images
        self.slice_axis = slice_axis

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if batch_idx % self.log_steps == 0:
            with torch.no_grad():
                inputs, labels = batch["img"], batch["seg"]

                # Prépare les figures pour affichage
                fig, axs = plt.subplots(self.num_images, 3, figsize=(15, 5 * self.num_images))

                for i in range(min(self.num_images, inputs.size(0))):
                    # Traitement des images d'entrée
                    input_image = inputs[i].cpu().numpy().squeeze()
                    label_image = labels[i].cpu().numpy().squeeze()

                    # Extraire la slice du milieu
                    mid_slice = input_image.shape[self.slice_axis] // 2
                    if self.slice_axis == 0:
                        input_slice = input_image[mid_slice, :, :]
                        label_slice = label_image[mid_slice, :, :]
                    elif self.slice_axis == 1:
                        input_slice = input_image[:, mid_slice, :]
                        label_slice = label_image[:, mid_slice, :]
                    else:
                        input_slice = input_image[:, :, mid_slice]
                        label_slice = label_image[:, :, mid_slice]

                    # Passage dans le modèle pour obtenir les sorties
                    output = pl_module(inputs[i].unsqueeze(0)).squeeze().cpu().numpy()
                    if self.slice_axis == 0:
                        output_slice = output[mid_slice, :, :]
                    elif self.slice_axis == 1:
                        output_slice = output[:, mid_slice, :]
                    else:
                        output_slice = output[:, :, mid_slice]

                    # Affichage des images d'origine et des sorties de modèle
                    axs[i, 0].imshow(input_slice, cmap='gray')
                    axs[i, 0].set_title(f'Original Image {i}')
                    axs[i, 0].axis('off')

                    axs[i, 1].imshow(label_slice, cmap='gray')
                    axs[i, 1].set_title(f'Ground Truth {i}')
                    axs[i, 1].axis('off')

                    axs[i, 2].imshow(output_slice, cmap='gray')
                    axs[i, 2].set_title(f'Model Output {i}')
                    axs[i, 2].axis('off')

                plt.tight_layout()

                # Vérifiez que le logger supporte l'upload des figures
                try:
                    run = trainer.logger.experiment
                    run["images/segmentation_comp"].upload(fig)
                except AttributeError as e:
                    print(f"Logging error: {e}")

                plt.close()