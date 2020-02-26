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
import cv2
from PIL import ImageFont
from PIL import Image
from utils import enhanceMorph
from glob import glob
os.environ["PYTHONBREAKPOINT"] = "pudb.set_trace"
from sklearn.metrics import roc_auc_score
if __name__ == "__main__":
    # Training settings
    
    model = Bottleneckv2(input_channels=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.load_state_dict(torch.load("weights/v2_420_MSE_512.pth", map_location=device))
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    # model = model.to(device)
    model.eval()

    for defect in ["crack", "cut", "hole", "print"]:
        files = glob(f"/Users/lehainam/AdvanceLab/image-anomaly-detection/dataset/mvtec_anomaly_detection/hazelnut/test/{defect}/*.png")
        total = 0
        for i, input_path in enumerate(files):
            gt_path = input_path.replace("test", "ground_truth").replace(".png","_mask.png")
            # input = cv2.imread(input_path)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            # cv2.namedWindow('image')
            # cv2.imshow('image',input)
            filename = os.path.basename(os.path.normpath(input_path))
            filename, ext = filename.split(".")

            input = Image.open(input_path)
            input= torchvision.transforms.ToTensor()(input).unsqueeze(0)
            # input_torch = torch.from_numpy(input).permute(2,0,1).unsqueeze(0)
            output = model(input)
            diff_avg = 1 - SSIM(input, output)[1]
            diff_avg = torch.mean(diff_avg, dim=1, keepdim=True)
            diff_avg = channelwised_normalize(diff_avg).detach()
            # enhanced_avg = enhanceMorph(diff_avg.numpy())
            
            folder = f"eval/test"
            # cv2.imwrite(f"{folder}/{filename}_gt.{ext}", gt)
            os.makedirs(folder, exist_ok=True)
            # torchvision.utils.save_image(diff_avg, f"{folder}/{filename}.{ext}")
            # torchvision.utils.save_image(enhanced_avg, f"{folder}/{filename}_enh.{ext}")

            gt= (gt/255.0).reshape(-1)
            diff_avg = diff_avg.numpy().squeeze(0).squeeze(0).reshape(-1)
            auc = roc_auc_score(gt, diff_avg)
            print(f"file {i}, {defect}: {auc}")
            total += auc
            # enhanced_avg = enhanced_avg.numpy().squeeze(0).squeeze(0).reshape(-1)
        print("*"*20)
        print(f"AVG ROC AUC: {defect} {total/i}")
    


    
    


