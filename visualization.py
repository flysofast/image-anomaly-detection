import argparse
import torch
from utils import channelwised_normalize, binarize, SSIM
import torch.nn as nn
import model as Model
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
import cv2
from PIL import ImageFont
from PIL import Image
from utils import enhanceMorph
from glob import glob
os.environ["PYTHONBREAKPOINT"] = "pudb.set_trace"
from sklearn.metrics import roc_auc_score
def gen_images():
    # Training settings
    saved_folder = "visualization"

    
    # m_paths = ["weights/v2_420_MSE_512.pth", "weights/v4_110_MSE_256.pth", "weights/v5_340_MSE_512.pth", "weights/v2_360_SSIM_512.pth", "weights/v4_380_SSIM_512.pth", "weights/v5_SSIM_450_256.pth"]
    # arcs = ["Bottleneckv2", "Bottleneckv4", "Bottleneckv5","Bottleneckv2", "Bottleneckv4", "Bottleneckv5"]
    m_paths = ["weights/v5_340_MSE_512.pth"]
    arcs = ["Bottleneckv5"]
    for model_path, model_arc in zip(m_paths, arcs):
        model = getattr(Model, model_arc)(input_channels = 3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model = model.to(device)
        model.eval()
        
        # for defect in ["crack", "cut", "hole", "print"]:
        for defect in ["hole"]:
            
            files = glob(f"dataset/mvtec_anomaly_detection/hazelnut/test/{defect}/*.png")
            # for i, input_path in enumerate(files):

            input_path = files[7]
            input_path = "dataset/mvtec_anomaly_detection/hazelnut/test/hole/006.png"
            gt_path = input_path.replace("test", "ground_truth").replace(".png","_mask.png")
            # input = cv2.imread(input_path)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            # cv2.namedWindow('image')
            # cv2.imshow('image',input)
            filename = os.path.basename(os.path.normpath(input_path))
            filename, ext = filename.split(".")

            input = Image.open(input_path)
            input= torchvision.transforms.ToTensor()(input).unsqueeze(0).to(device)
            # input_torch = torch.from_numpy(input).permute(2,0,1).unsqueeze(0)
            output = model(input)
            diff_avg = 1 - SSIM(input, output)[1]
            diff_avg = torch.mean(diff_avg, dim=1, keepdim=True)
            diff_avg = channelwised_normalize(diff_avg).detach().cpu()
            th_diff, gth_diff, otsu_diff = binarize(diff_avg, output_channels=3)
                
            # enhanced_avg = enhanceMorph(diff_avg.numpy())
            
            folder = f"{saved_folder}/{defect}"
            cv2.imwrite(f"{folder}/{filename}_gt.{ext}", gt)
            os.makedirs(folder, exist_ok=True)
            torchvision.utils.save_image(input, f"{folder}/{filename}_input.{ext}")
            torchvision.utils.save_image(output, f"{folder}/{filename}_output.{ext}")
            torchvision.utils.save_image(diff_avg, f"{folder}/{filename}_res.{ext}")
            torchvision.utils.save_image(th_diff, f"{folder}/{filename}_bin.{ext}")
            torchvision.utils.save_image(gth_diff, f"{folder}/{filename}_gaus.{ext}")
            torchvision.utils.save_image(otsu_diff, f"{folder}/{filename}_otsu.{ext}")
            print("saved")
            # torchvision.utils.save_image(enhanced_avg, f"{folder}/{filename}_enh.{ext}")
def get_masked_image(background, mask,channel=0):
    background = np.asarray(background).copy()
    mask = np.asarray(mask).copy()
    w= mask.shape[1]
    mask = mask[:, w//2:]
    indices = mask != 0
    for i in range(3):
        val = 255 if i==channel else 0
        background[:,:,i][indices] = val
    return Image.fromarray(background)

if __name__ == "__main__":
    gen_images()
    
    input= "visualization/hole/006_input.png"
    gt = input.replace("_input","_gt")
    # input ="print.png"
    pre = input.replace("_input","_otsu")
    background = Image.open(input)
    overlay = Image.open(pre).convert("L")
   
    overlay = get_masked_image(background, overlay, channel=1)

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    new_img1 = Image.blend(background, overlay, 0.6)
    new_img1.save("hole.png","PNG")

    # background = Image.open(input)
    # overlay = Image.open(pre)
   
    # overlay = get_masked_image(new_img1, overlay, channel=1)
   
    # background = new_img1.convert("RGBA")
    # overlay = overlay.convert("RGBA")
    # new_img2 = Image.blend(background, overlay, 0.5)




               