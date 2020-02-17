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
import cv2
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

    with torch.no_grad():
        for index, (img, gt) in enumerate(tqdm(test_loader)):
            # img, _ = data
            img = img.to(device)
            gt = gt.to(device)
            output = model(img)
            diff = torch.abs(img - output)
            
            # Grayscaled diff image (average of 3 channels)
            diff_avg = torch.mean(diff, dim=1).unsqueeze(1)

            diff_avg = channelwised_normalize(diff_avg)
            diff = channelwised_normalize(diff)     
            
            th_diff, gth_diff, otsu_diff = binarize(diff_avg)

            # Make the grayscale image 3-channeled
            diff_avg = diff_avg.expand(-1, 3, -1, -1)
            loss = loss_op(diff_avg, gt)
            test_epoch_loss += loss.item()

            # Save the results if requested
            if index in chosen_sample_i:
                diff_avg = diff_avg
                image = torch.cat((img, output, gt, diff_avg, th_diff, gth_diff, otsu_diff), 0)
                if test_images is None:
                    test_images = image
                else:
                    test_images = torch.cat((test_images, image), dim=0)

        test_epoch_loss = test_epoch_loss/len(test_loader)
        test_images = torchvision.utils.make_grid(test_images, nrow=7)
        result_image = os.path.join(results_folder, f"val_{epoch}.png")
        torchvision.utils.save_image(test_images, result_image)
        print(f"Test images saved at {results_folder}")
        
         # write to tensorboard
        if writer:
            writer.add_image('Test images', test_images, global_step=epoch)
            writer.add_scalar("MSE/segmentation_test", test_epoch_loss, global_step=epoch )
    return test_epoch_loss

def channelwised_normalize(batch_image):
    """
    Normalize the batch of images by scaling the values in each channel of each image, so that the [min, max] range of each channel will be mapped to [0, 1]
    
    Returns:
        Normalized image batch
    """
    for si in range(batch_image.shape[0]):
        for ci in range(batch_image.shape[1]):
            channel = batch_image[si, ci,:,:]
            mi = torch.min(channel)
            ma = torch.max(channel)
            batch_image[si, ci,:,:] = (channel - mi)/(ma - mi)
    return batch_image

def binarize(batch_image):
    """
        Binarize the input image with binary threshold, using 0.5 threshold, Gaussian adaptive threshold and OTSU threshold
        Parameters:
            batch_image: tensor batch of images (usually of 1)
        Returns:
            3 thresholded images
    """
    th_batch, gth_batch, otsu_batch = torch.zeros_like(batch_image), torch.zeros_like(batch_image), torch.zeros_like(batch_image)
    batch_image = batch_image.permute(0, 2, 3, 1).cpu().numpy()
    for i, image in enumerate(batch_image):
        img = cv2.convertScaleAbs(image, 0 , 255)

        _ ,img1 = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
        cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        img2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        _ ,img3 = cv2.threshold(img,128,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.normalize(img3, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        th_batch[i] = torch.from_numpy(img1).unsqueeze(2).permute(2,0,1)
        gth_batch[i] = torch.from_numpy(img2).unsqueeze(2).permute(2,0,1)
        otsu_batch[i] = torch.from_numpy(img3).unsqueeze(2).permute(2,0,1)
    return th_batch.expand(-1, 3, -1, -1), gth_batch.expand(-1, 3, -1, -1), otsu_batch.expand(-1, 3, -1, -1)
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Autoencoder anomaly detection')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--weights_path', type=str, default="BottleNeckv3_130.pth", metavar='w',
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
    test_on_mixed_samples(model=model, test_loader=test_loader, loss_op=nn.MSELoss(), writer=None, results_folder="test_results", n_saved_results=5)