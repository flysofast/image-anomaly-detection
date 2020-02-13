import argparse
import torch
import torch.nn as nn
from model import Autoencoder
import numpy as np
from tqdm import tqdm
import random
import torchvision
from tensorboardX import SummaryWriter
import os
def test_on_mixed_samples(model, test_loader, loss_op, epoch, writer, results_folder, saving = True, n_saved_results=5):
    print("Testing on mixed set...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_epoch_loss = 0
    val_images = None
    if len(test_loader) > n_saved_results:
        chosen_sample_i = torch.multinomial(torch.Tensor(range(len(test_loader))), num_samples = n_saved_results, replacement=False)
    else:
        chosen_sample_i = range(len(test_loader))

    with torch.no_grad():
        for index, (img,_) in enumerate(tqdm(test_loader)):
            # img, _ = data
            img = img.to(device)
            output = model(img)
            loss = loss_op(output, img)
            val_epoch_loss += loss.item()
            
            if saving and index in chosen_sample_i:
                diff = torch.abs(img - output)

                # Grayscaled diff image (average of 3 channels)
                diff_avg = torch.mean(diff, dim=1).unsqueeze(1).expand(-1, 3, -1, -1)
                
                image = torch.cat((img, output, diff, diff_avg), 0)
                if val_images is None:
                    val_images = image
                else:
                    val_images = torch.cat((val_images, image), dim=0)

        # write to tensorboard
        writer.add_scalar("MSE/test", val_epoch_loss/len(test_loader), global_step=epoch )
        if saving:
            val_images = torchvision.utils.make_grid(val_images, nrow=4)
            writer.add_image('Test images', val_images, global_step=epoch)

            result_image = os.path.join(results_folder, f"val_{epoch}.png")
            torchvision.utils.save_image(val_images, result_image)
            print(f"Test images saved at {results_folder}")

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Autoencoder anomaly detection')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--weights_path', type=str, default="output/AE_10:01AM on February 13, 2020/models/model_3.pth", metavar='w',
                        help='Saving interval in number of epochs')

    args = parser.parse_args()
    model = Autoencoder()
    model.load_state_dict(torch.load(args.model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    