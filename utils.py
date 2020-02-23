
import torch
import cv2
import numpy as np
# from skimage.metrics import structural_similarity as ssim
import skimage

import math
from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# def ssim_compare(img1, img2, win=11):
#     # padding = win//2

#     # img1 = np.pad(img1.cpu().numpy(), padding, mode="symmetric")
#     # img2 = np.pad(img2.cpu().numpy(), padding, mode="symmetric")
    
#     return skimage.measure.com

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def SSIM(img1, img2):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(), ssim_map


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

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

def binarize(batch_image, output_channels):
    """
        Binarize the input image with binary threshold, using 0.5 threshold, Gaussian adaptive threshold and OTSU threshold
        Parameters:
            batch_image: tensor batch of images (usually of 1)
        Returns:
            3 thresholded images
    """
    output_shape = (batch_image.shape[0], batch_image.shape[1], batch_image.shape[2], batch_image.shape[3]*2)
    th_batch, gth_batch, otsu_batch = torch.zeros(output_shape), torch.zeros(output_shape), torch.zeros(output_shape)
    batch_image = batch_image.permute(0, 2, 3, 1).cpu().numpy()
    for i, image in enumerate(batch_image):
        img = cv2.convertScaleAbs(image) * 255

        _ ,img1 = cv2.threshold(img,20,255,cv2.THRESH_BINARY)
        img1 = np.hstack((img1, enhanceMorph(img1)))
        cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        img2 = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
        img2 = np.hstack((img2, enhanceMorph(img2)))
        cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        _ ,img3 = cv2.threshold(img,128,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img3 = np.hstack((img3, enhanceMorph(img3)))
        cv2.normalize(img3, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        th_batch[i] = torch.from_numpy(img1).unsqueeze(2).permute(2,0,1)
        gth_batch[i] = torch.from_numpy(img2).unsqueeze(2).permute(2,0,1)
        otsu_batch[i] = torch.from_numpy(img3).unsqueeze(2).permute(2,0,1)
    return th_batch.expand(-1, output_channels, -1, -1), gth_batch.expand(-1, output_channels, -1, -1), otsu_batch.expand(-1, output_channels, -1, -1)

def enhanceMorph(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    img = cv2.erode(img, kernel, iterations=1) # Clean noise pixels
    img = cv2.dilate(img, kernel, iterations=3) # Connect segments
    img = cv2.erode(img, kernel, iterations=1) # Remove noises
    return img


# if __name__ == "__main__":
    
#     a,b = SSIM(torch.rand(1,3,100, 100), torch.rand(1,3,100, 100))
#     print(a)
#     print(b.shape)