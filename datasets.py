import os
import pandas as pd
# from skimage import io, transform
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from PIL import Image
import numpy as np
from glob import glob
import torchvision.transforms as transforms
import torch
# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

class MVTecAd(Dataset):
    """MVTecAd dataset."""

    def __init__(self, subset="train", category="hazelnut", root_dir="dataset/mvtec_anomaly_detection", val_split= "val_split.txt", transform=None):
        """
        Args:
            subset: can be "train" or "test", which represents the orignal splits of the category from the dataset
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert subset in ["train", "test"], "Invalid subset name"

        # test_split =  open(os.path.join(root_dir, "train_test_split.txt"))
        # test_set = [path for path in test_split.read().splitlines() if os.path.join(root_dir, category) in path]
        # if subset == "test":
        #     good_images = [(img, 1) for img in glob(os.path.join(root_dir, category, "test", "good", "*.png"))]
            
        # else:
            
        #     self.images = [ (img, 1) for img in glob(os.path.join(root_dir, category, "train", "**", "*.png"))]
        self.images = glob(os.path.join(root_dir, category, subset, "**", "*.png"))
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset
        self.category = category

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            img: the image samples
            gt: the ground truth for defectious samples. If the samples is good the return an all-zeros mask
        """
        img_path = self.images[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        
        gt = torch.zeros_like(img)
        # Load ground truth if it's available
        if self.subset == "test":
            gt_path = img_path.replace(os.path.join(self.category, self.subset), os.path.join(self.category, "ground_truth"))

            # Deal with filenames only to be more robust in case of weird folder names
            fn = os.path.basename(os.path.normpath(gt_path))
            new_fn = fn.replace(".png","_mask.png")
            gt_path = gt_path.replace(fn, new_fn)
            # Good samples won't have ground truth
            if os.path.exists(gt_path):
                gt = Image.open(gt_path)
                gt = transforms.functional.to_tensor(gt).expand(3, -1, -1)
        
        return img, gt

class HAM10000(Dataset):
    """HAM10000 dataset."""

    def __init__(self, subset="train", csv_file="HAM10000_metadata.csv", image_folder="HAM10000_images", root_dir="dataset",  transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert subset in ["train", "test"], "Invalid subset name"
        csv_file = os.path.join(root_dir, csv_file)
        print(f"Reading annotations at {csv_file}...")

        df = pd.read_csv(csv_file, header=0, usecols = ["lesion_id", "image_id", "dx"])
        mel_samples = df.dx == "mel"

        if(subset == "train"):
            self.annotations = df[~mel_samples]
        elif subset == "test":
            self.annotations = df[mel_samples]

        print("Done.")
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = image_folder

    def __len__(self):
        return int(len(self.annotations)/100)

    def __getitem__(self, idx):
        lesion = self.annotations.iloc[idx]

        img_name = os.path.join(self.root_dir, self.image_folder,
                                f"{lesion['image_id']}.jpg")
        img = Image.open(img_name)
        label = lesion["dx"]
        if self.transform:
            img = self.transform(img)
        img = img

        return img, label

def get_trainval_samplers(dataset: Dataset, validation_split = 0.2):
    """
        Creates random samples for train/val splits
    """

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler

if __name__ == "__main__":
    data_transform = transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomPerspective(),
        transforms.RandomCrop(400),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
    ])
    dataset = MVTecAd(subset="test", category="hazelnut", root_dir="dataset/mvtec_anomaly_detection", transform=data_transform)
    trainloader = DataLoader(dataset, batch_size=1, num_workers=4)
    for img, gt in trainloader:
        print(img.shape)
