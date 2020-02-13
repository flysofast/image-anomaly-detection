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
            subset: can be "train", "val" or "test". "train" and "test" are the original splits substract the items in the "val" set, which were predefined and contains both classes
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert subset in ["train", "val", "test"], "Invalid subset name"

        val_split =  open(os.path.join(root_dir, "val_split.txt"))
        val_set = [path for path in val_split.read().splitlines() if os.path.join(root_dir, category) in path]
        if subset == "val":
            self.images = val_set
        else:
            self.images = [img for img in glob(os.path.join(root_dir, category, subset, "**", "*.png")) if img not in val_set]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # return 5
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        
        return img

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

def get_trainval_samplers(dataset: Dataset, validation_split = 0.2, random_seed=42):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(random_seed)
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
