import os
import pandas as pd
# from skimage import io, transform
from torch.utils.data import Dataset, SubsetRandomSampler
from PIL import Image
import numpy as np
# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")


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
