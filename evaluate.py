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
    test_images2 = None
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
            # diff = torch.abs(img - output)
            
            # Grayscaled diff image (average of 3 channels)
            diff_avg = 1 - SSIM(img, output)[1]
            diff_avg = torch.mean(diff_avg, dim=1, keepdim=True)
            diff_avg = channelwised_normalize(diff_avg)
            # diff = channelwised_normalize(diff)     
            th_diff, gth_diff, otsu_diff = binarize(diff_avg, n_output_channels)

            # Make the grayscale image 3-channeled
            # diff_avg = diff_avg
            loss = 1-loss_op(diff_avg, gt)
            test_epoch_loss += loss.item()

            # Save the results if requested
            if index in chosen_sample_i:
                io_pair = torch.cat((img, output), dim=3)
                gt_pair = torch.cat((gt, diff_avg), dim=3)
                gt_pair = gt_pair.squeeze(0)
                gt_pair = transforms.ToPILImage()(gt_pair.cpu())
                draw = ImageDraw.Draw(gt_pair)
                font = ImageFont.truetype(font="BebasNeue-Regular.ttf", size=150)
                # font = ImageFont.truetype("sans-serif.ttf", 16)

                draw.text((0,0),f"{loss.item():.3f}", (0), font=font)
                draw.text((0,25),f"{loss.item():.3f}",(255), font=font)
                gt_pair = transforms.ToTensor()(gt_pair).unsqueeze(0).expand(-1, n_output_channels, -1, -1).to(device)
                image = torch.cat((io_pair.to(device), gt_pair.to(device), th_diff.to(device), gth_diff.to(device), otsu_diff.to(device)), 0)
                if test_images is None:
                    test_images = image
                else:
                    test_images = torch.cat((test_images, image), dim=0)
                
                #####DIRTY ADDITION DIFF MAP RESULT#######
                diff = torch.abs(img - output)
                # Grayscaled diff image (average of 3 channels)
                diff_avg = torch.mean(diff, dim=1, keepdim=True)
                diff_avg = channelwised_normalize(diff_avg)
                # diff = channelwised_normalize(diff)     
                th_diff, gth_diff, otsu_diff = binarize(diff_avg, n_output_channels)

                # Make the grayscale image 3-channeled
                # diff_avg = diff_avg
                loss = nn.MSELoss()(diff_avg, gt)

                # Save the results if requested
                if index in chosen_sample_i:
                    gt_pair = torch.cat((gt, diff_avg), dim=3)
                    gt_pair = gt_pair.squeeze(0)
                    gt_pair = transforms.ToPILImage()(gt_pair.cpu())
                    draw = ImageDraw.Draw(gt_pair)
                    font = ImageFont.truetype(font="BebasNeue-Regular.ttf", size=150)
                    # font = ImageFont.truetype("sans-serif.ttf", 16)

                    draw.text((0,0),f"{loss.item():.3f}", (0), font=font)
                    draw.text((0,25),f"{loss.item():.3f}",(255), font=font)
                    gt_pair = transforms.ToTensor()(gt_pair).unsqueeze(0).expand(-1, n_output_channels, -1, -1).to(device)
                    image = torch.cat((io_pair.to(device), gt_pair.to(device), th_diff.to(device), gth_diff.to(device), otsu_diff.to(device)), 0)
                    if test_images2 is None:
                        test_images2 = image
                    else:
                        test_images2 = torch.cat((test_images2, image), dim=0)
                    

        test_epoch_loss = test_epoch_loss/len(test_loader)
        test_images = torchvision.utils.make_grid(test_images, nrow=5)
        test_images = test_images.unsqueeze(0)
        test_images = F.interpolate(test_images, scale_factor=0.1)
        result_image = os.path.join(results_folder, f"val_{epoch}.png")
        torchvision.utils.save_image(test_images, result_image)
        print(f"Test images saved at {results_folder}")

        test_images2 = torchvision.utils.make_grid(test_images2, nrow=5)
        test_images2 = test_images2.unsqueeze(0)
        test_images2 = F.interpolate(test_images2, scale_factor=0.1)
        result_image = os.path.join(results_folder, f"val_{epoch}_more.png")
        torchvision.utils.save_image(test_images2, result_image)
        print(f"Additional diff map images saved at {results_folder}")
    
         # write to tensorboard
        if writer:
            test_images = test_images.squeeze(0)
            writer.add_image('Test images', test_images, global_step=epoch)
            writer.add_scalar(f"{loss_op.__class__.__name__}/Test", test_epoch_loss, global_step=epoch )

    return test_epoch_loss

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Autoencoder anomaly detection')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--weights_path', type=str, default="output/BottleNeckv_5_09:46PM on February 24, 2020/models/Bottleneckv5_0.pth", metavar='w',
                        help='Saving interval in number of epochs')

    args = parser.parse_args()
    model = Bottleneckv5()
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