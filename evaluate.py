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
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import MVTecAd
def test_on_mixed_samples(model, test_loader, loss_op, writer, results_folder, saving = True, n_saved_results=5, epoch=0):
    """
        Perform evaluation on the test set
        Returns: average MSE error on the whole test set
    """
    print("Testing on mixed set...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_epoch_loss = 0
    test_images = None
    if len(test_loader) > n_saved_results:
        chosen_sample_i = torch.multinomial(torch.Tensor(range(len(test_loader))), num_samples = n_saved_results, replacement=False)
    else:
        chosen_sample_i = range(len(test_loader))

    with torch.no_grad():
        for index, (img, gt) in enumerate(tqdm(test_loader)):
            # img, _ = data
            img = img.to(device)
            output = model(img)
            diff = torch.abs(img - output)
            
            # Grayscaled diff image (average of 3 channels)
            diff_avg = torch.mean(diff, dim=1).unsqueeze(1)

            diff_avg = channelwised_normalize(diff_avg)
            diff = channelwised_normalize(diff)     

            # # Ignore good samples for the error calculation
            # if type(gt) is int and gt == 0:
            #     loss = loss_op(diff_avg, gt)
            #     test_epoch_loss += loss.item()

            # Save the results if requested
            if saving and index in chosen_sample_i:
                # Make the grayscale image 3-channeled
                diff_avg = diff_avg.expand(-1, 3, -1, -1)
                image = torch.cat((img, output, gt, diff, diff_avg), 0)
                if test_images is None:
                    test_images = image
                else:
                    test_images = torch.cat((test_images, image), dim=0)

        test_epoch_loss = test_epoch_loss/len(test_loader)
        # write to tensorboard
        if writer:
            writer.add_scalar("MSE/segmentation_test", test_epoch_loss, global_step=epoch )
        if saving:
            test_images = torchvision.utils.make_grid(test_images, nrow=5)

            if writer:
                writer.add_image('Test images', test_images, global_step=epoch)

            result_image = os.path.join(results_folder, f"val_{epoch}.png")
            torchvision.utils.save_image(test_images, result_image)
            print(f"Test images saved at {results_folder}")
    return test_epoch_loss

def channelwised_normalize(batch_image):
    for si in range(batch_image.shape[0]):
        for ci in range(batch_image.shape[1]):
            channel = batch_image[si, ci,:,:]
            mi = torch.min(channel)
            ma = torch.max(channel)
            batch_image[si, ci,:,:] = (channel - mi)/(ma - mi)
    return batch_image


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Autoencoder anomaly detection')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--weights_path', type=str, default="model_860.pth", metavar='w',
                        help='Saving interval in number of epochs')

    args = parser.parse_args()
    model = Autoencoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    test_data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    testset = MVTecAd(subset="test", category="hazelnut", root_dir="dataset/mvtec_anomaly_detection", transform=test_data_transform)
    test_loader = DataLoader(testset, batch_size=1, num_workers=4, shuffle=True)
    test_on_mixed_samples(model=model, test_loader=test_loader, loss_op=nn.MSELoss(), writer=None, results_folder="test_results", saving=True, n_saved_results=5)