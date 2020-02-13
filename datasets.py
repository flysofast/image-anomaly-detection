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

    def __len__(self):
        # return 10
        return len(self.images)

    def __getitem__(self, idx):
        # path, label = self.images[idx].split(",")
        # label = int(label)
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        
        return img, 1 #labels are ignored for now

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
    for img in trainloader:
        print(img.shape)
