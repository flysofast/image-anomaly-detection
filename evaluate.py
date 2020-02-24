import argparse
import torch
from utils import channelwised_normalize, binarize, SSIM
import torch.nn as nn
from model import *
import numpy as np
from tqdm import tqdm
import random
import torchvision
from tensorboardX import SummaryWriter
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from datasets import MVTecAd
from matplotlib import pyplot as plt
from PIL import ImageDraw
from PIL import ImageFont
os.environ["PYTHONBREAKPOINT"] = "pudb.set_trace"
def test_on_mixed_samples(model, test_loader, loss_op, writer, results_folder, n_saved_results=5, epoch=0):
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
    n_output_channels = 3
    with torch.no_grad():
        for index, (img, gt) in enumerate(tqdm(test_loader)):
            # img, _ = data
            img = img.to(device)
            n_output_channels = img.shape[1]
            gt = gt.to(device)
            output = model(img)
            diff = torch.abs(img - output)
            
            # Grayscaled diff image (average of 3 channels)
            _, diff_avg = SSIM(img, output)
            diff_avg = torch.mean(diff, dim=1).unsqueeze(1)

            diff_avg = channelwised_normalize(diff_avg)
            diff = channelwised_normalize(diff)     
            th_diff, gth_diff, otsu_diff = binarize(diff_avg, n_output_channels)

            # Make the grayscale image 3-channeled
            diff_avg = diff_avg.expand(-1, n_output_channels, -1, -1)
            loss = loss_op(diff_avg, gt)
            test_epoch_loss += loss.item()

            # Save the results if requested
            if index in chosen_sample_i:
                io_pair = torch.cat((img, output), dim=3)
                gt_pair = torch.cat((gt, diff_avg), dim=3)
                
                image = torch.cat((io_pair, gt_pair, th_diff, gth_diff, otsu_diff), 0)
                if test_images is None:
                    test_images = image
                else:
                    test_images = torch.cat((test_images, image), dim=0)
                
                # test_images = transforms.ToPILImage()(test_images)
                # draw = ImageDraw.Draw(test_images)
                # font = ImageFont.truetype(font="BebasNeue-Regular.ttf", size=23)

                # draw.text(
                #     (0,0),
                #     f"{loss.item():.3f}",
                #     (0,0,0), font=font
                # )
                # draw.text(
                #     (0,25),
                #     f"{loss.item():.3f}",
                #     (255,255,255), font=font
                # )
                # test_images = transforms.ToTensor()(test_images).unsqueeze(0)

        test_epoch_loss = test_epoch_loss/len(test_loader)
        test_images = torchvision.utils.make_grid(test_images, nrow=5)
        test_images = F.interpolate(test_images, scale_factor=1/5)
        result_image = os.path.join(results_folder, f"val_{epoch}.png")
        torchvision.utils.save_image(test_images, result_image)
        print(f"Test images saved at {results_folder}")
    
         # write to tensorboard
        if writer:
            test_images = test_images.squeeze(0)
            writer.add_image('Test images', test_images, global_step=epoch)
            writer.add_scalar("MSE/segmentation_test", test_epoch_loss, global_step=epoch )

    return test_epoch_loss




if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Autoencoder anomaly detection')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--weights_path', type=str, default="weights/BottleNeckv4_50.pth", metavar='w',
                        help='Saving interval in number of epochs')

    args = parser.parse_args()
    model = Bottleneckv4()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    test_data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    testset = MVTecAd(subset="test", category="zipper", root_dir="dataset/mvtec_anomaly_detection", transform=test_data_transform)
    test_loader = DataLoader(testset, batch_size=1, num_workers=4, shuffle=True)
    test_on_mixed_samples(model=model, test_loader=test_loader, loss_op=nn.MSELoss(), writer=None, results_folder="test_results", n_saved_results=5)