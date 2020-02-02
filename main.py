import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import os
from dataset import HAM10000, get_trainval_samplers
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from model import Autoencoder
os.environ["PYTHONBREAKPOINT"] = "pudb.set_trace"
# Loading and Transforming data
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

dataset = HAM10000(subset="train", csv_file="HAM10000_metadata.csv", image_folder="HAM10000_images", root_dir="dataset", transform=data_transform)
# train_sampler, val_sampler = get_trainval_samplers(dataset, validation_split=0.2)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, sampler=val_sampler)
# valset = HAM10000(subset="test", csv_file="HAM10000_metadata.csv", image_folder="HAM10000_images", root_dir="dataset", transform=data_transform)
# val_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

num_epochs = 200 
batch_size = 128

model = Autoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Using", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model = model.to(device)

loss_op = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)
weights_folder = "output"
os.makedirs(weights_folder, exist_ok = True)

print("Start training")
for epoch in range(num_epochs):
    model.train(True)
    pbar = tqdm(train_loader)
    pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

    for (img, label) in pbar:
        img = img.to(device)
        optimizer.zero_grad()
        
        # ===================forward=====================
        output = model(img)
        loss = loss_op(output, img)
        # ===================backward====================
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data.cpu().numpy()))
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(weights_folder, f"model_{epoch}.pth"))

    # print("Validating")
    # model.eval()
    # with torch.no_grad():
    #     for index, (img, label) in enumerate(tqdm(val_loader)):
    #         # img, _ = data
    #         img = img.to(device)
    #         output = model(img)
    #         output = output.detach()
    #         image = torch.cat((img, output), 2)
    #         image = image.cpu()
    #         save_image(image,os.path.join(weights_folder, f"output_{epoch}.png") )

