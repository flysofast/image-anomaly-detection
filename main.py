import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
from datasets import HAM10000, get_trainval_samplers
# import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from model import *
from datasets import MVTecAd
import datetime
from tensorboardX import SummaryWriter
import argparse
from evaluate import test_on_mixed_samples
from splitter import split_train_test
import torchsummary
os.environ["PYTHONBREAKPOINT"] = "pudb.set_trace"

def train(train_loader, val_loader, test_loader, args):
    num_epochs = args.epochs

    model = Bottleneckv2()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    loss_op = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=args.lr)
    exp_name = f'{args.exp_name}_{datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")}'
    output_folder = os.path.join("output", exp_name)
    weights_folder = os.path.join(output_folder, "models")
    results_folder = os.path.join(output_folder, "results")
    os.makedirs(weights_folder, exist_ok = True)
    os.makedirs(results_folder, exist_ok = True)

    writer = SummaryWriter(log_dir=f"log/{exp_name}")
    print("Start training")
    min_loss = 1e10
    # saving = False
    for epoch in range(num_epochs):
        print(f"==============================Epoch {epoch+1}/{num_epochs}==============================")
        model.train(True)
        train_batches = tqdm(train_loader)
        epoch_loss = 0
        for i, (img,_) in enumerate(train_batches):
            if epoch == 0 and i == 0:
                torchsummary.summary(model, img[0].shape)

            img = img.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_op(output, img)

            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            epoch_loss += loss_val
            writer.add_scalar("MSE/train", loss_val, epoch * len(train_loader) + i)
        train_loss = epoch_loss/len(train_loader)
        
        #=================Validate the autoencoder on val set===============
        print("Validating...")
        val_batches = tqdm(val_loader)
        model.eval()
        epoch_loss = 0
        for i, (img, _) in enumerate(val_batches):
            
            img = img.to(device)
            output = model(img)
            loss = loss_op(output, img)

            loss_val = loss.item()
            epoch_loss += loss_val
            writer.add_scalar("MSE/val", loss_val, epoch * len(val_loader) + i)

        #========Test on mixed set=============
        test_error = test_on_mixed_samples(model=model, test_loader=test_loader, 
                            loss_op=loss_op, epoch=epoch, writer=writer, results_folder=results_folder, 
                             n_saved_results=args.n_saved_samples)
        print(f'Epoch [{epoch+1}/{num_epochs}], train_loss:{train_loss:.4f} val_loss:{epoch_loss/len(val_loader):.4f} test error: {test_error:.4f}')
        
        if test_error < min_loss or epoch % args.save_interval == 0: 
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            model_file = os.path.join(weights_folder, f"{model.__class__.__name__}_{epoch}.pth")
            torch.save(state_dict, model_file)
            print(f"Model saved at {model_file}")
        
        if test_error < min_loss:
            min_loss = test_error


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Autoencoder anomaly detection')
    parser.add_argument('--exp_name', type=str, default="AE")
    parser.add_argument('--batch_size', type=int, default=32, metavar='b',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200, metavar='ne',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--num_workers', type=float, default=4)
    # parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')

    # Data settings
    parser.add_argument('--category', type=str, default="hazelnut")

    parser.add_argument('--crop_size', type=int, default=400, metavar='cs',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--n_saved_samples', type=int, default=5, help='number of random saved samples during testing (default: 5)')
    parser.add_argument('--save_interval', type=int, default=10, metavar='i',
                        help='Saving interval in number of epochs')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    # Loading and Transforming data
    train_data_transform = transforms.Compose([
            # transforms.Resize(512),
            # transforms.RandomSizedCrop(224),
            # transforms.RandomPerspective(),
            transforms.RandomCrop(args.crop_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
        ])
    test_data_transform = transforms.Compose([
            # transforms.Resize(512),
            transforms.ToTensor(),
        ])

    testset = MVTecAd(subset="test", category=args.category, root_dir="dataset/mvtec_anomaly_detection", transform=test_data_transform)
    test_loader = DataLoader(testset, batch_size=1, num_workers=4, shuffle=True)

    trainset = MVTecAd(subset="train", category=args.category, root_dir="dataset/mvtec_anomaly_detection", transform=train_data_transform)
    ts, vs = get_trainval_samplers(trainset, validation_split=0.2)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=ts)
    val_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=vs)
    train(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, args=args)
    


