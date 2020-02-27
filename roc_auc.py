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
if __name__ == "__main__":
    # Training settings
    eval_folder = "eval"
    os.makedirs(eval_folder, exist_ok=True)
    log = open(os.path.join(eval_folder, "benchmark.txt"), "w")

    perf_results = {}
    min_cat = {"auc": 2}
    max_cat = {"auc": 0}
    max_all = {"auc": 0}
    min_all = {"auc": 2}
    m_paths = ["weights/v2_420_MSE_512.pth", "weights/v4_110_MSE_256.pth", "weights/v5_400_MSE_512", "weights/v2_360_SSIM_512.pth", "weights/v4_380_SSIM_512.pth", "weights/v5_SSIM_450_256.pth"]
    arcs = ["Bottleneckv2", "Bottleneckv4", "Bottleneckv5","Bottleneckv2", "Bottleneckv4", "Bottleneckv5"]
    for model_path, model_arc in zip(m_paths, arcs):
        print(f"{'='*30}{model_path}{'='*30}")
        log.write(f"{'='*30}{model_path}{'='*30}\n")
        model = getattr(Model, model_arc)(input_channels = 3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model = model.to(device)
        model.eval()
        perf_results[model_path] = {
            "weight" : model_path,
            # "mean_all": 0,
            "performance": [
            #     {
            #         "defect_type": "",
            #         "mean_auc": 0,
            #         "details": []
            #     }
            ]
        }
        overall_total = 0
        overall_files_count = 0
        for defect in ["crack", "cut", "hole", "print"]:
            performance = {
                    "defect_type": defect,
                    "mean_auc": 0,
                    "details": []
                }
            print(f"{defect}:")
            log.write(f"{defect}:\n")
            files = glob(f"dataset/mvtec_anomaly_detection/hazelnut/test/{defect}/*.png")
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
                input= torchvision.transforms.ToTensor()(input).unsqueeze(0).to(device)
                # input_torch = torch.from_numpy(input).permute(2,0,1).unsqueeze(0)
                output = model(input)
                diff_avg = 1 - SSIM(input, output)[1]
                diff_avg = torch.mean(diff_avg, dim=1, keepdim=True)
                diff_avg = channelwised_normalize(diff_avg).detach().cpu()
                # enhanced_avg = enhanceMorph(diff_avg.numpy())
                
                # folder = f"{eval_folder}/{defect}"
                # cv2.imwrite(f"{folder}/{filename}_gt.{ext}", gt)
                # os.makedirs(folder, exist_ok=True)
                # torchvision.utils.save_image(diff_avg, f"{folder}/{filename}.{ext}")
                # torchvision.utils.save_image(enhanced_avg, f"{folder}/{filename}_enh.{ext}")

                gt= (gt/255.0).reshape(-1)
                diff_avg = diff_avg.numpy().squeeze(0).squeeze(0).reshape(-1)
                auc = roc_auc_score(gt, diff_avg)
                print(f"file {i}, {defect}: {auc}")
                log.write(f"file {i}, {defect}: {auc}\n")
                total += auc
                performance["details"].append({
                    "file_path": input_path,
                    "auc": auc
                })
                # enhanced_avg = enhanced_avg.numpy().squeeze(0).squeeze(0).reshape(-1)
            mean_auc = total/(len(files))
            print("-"*20)
            log.write("-"*20)
            log.write("\n")
            print(f"Average AUC: {defect} - {mean_auc} over {len(files)} files.")
            log.write(f"Average AUC: {defect} - {mean_auc} over {len(files)} files. \n")
            performance["mean_auc"] = mean_auc
            if mean_auc > max_cat["auc"]:
                max_cat["auc"] = mean_auc
                max_cat["model"] = model_path
                max_cat["cat"] = defect
            if mean_auc > min_cat["auc"]:
                min_cat["auc"] = mean_auc
                min_cat["model"] = model_path
                min_cat["cat"] = defect
            
            perf_results[model_path]["performance"].append(performance)
            overall_total+=total
            overall_files_count += len(files)
        
        mean_all = overall_total/overall_files_count
        perf_results[model_path]["mean_all"] = mean_all
        print(f"Average overall: {mean_all}")
        log.write(f"Average overall: {mean_all}\n")
        if mean_all > max_all["auc"]:
                max_all["auc"] = mean_all
                max_all["model"] = model_path
        if mean_all > min_all["auc"]:
                min_all["auc"] = mean_auc
                min_all["model"] = model_path
    
    np.save(os.path.join(eval_folder, f"roc_benchmark.npy"), perf_results)

    print(f"-----Max:-----")
    print(f"By category: {max_cat['cat']} {max_cat['auc']} - {max_cat['model']}")
    print(f"Overall: {max_all['auc']} - {max_all['model']}")
    print(f"-----Min:-----")
    print(f"By category: {min_cat['cat']} {min_cat['auc']} - {min_cat['model']}")
    print(f"Overall: {min_all['auc']} - {min_all['model']}")

    log.write(f"-----Max:-----\n")
    log.write(f"By category: {max_cat['cat']} {max_cat['auc']} - {max_cat['model']}\n")
    log.write(f"Overall: {max_all['auc']} - {max_all['model']}\n")
    log.write(f"-----Min:-----\n")
    log.write(f"By category: {min_cat['cat']} {min_cat['auc']} - {min_cat['model']}\n")
    log.write(f"Overall: {min_all['auc']} - {min_all['model']}\n")
    log.close()





    
    


