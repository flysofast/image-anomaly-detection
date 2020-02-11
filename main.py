import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import os
from datasets import HAM10000, get_trainval_samplers
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from model import Autoencoder
from datasets import MVTecAd
import datetime
os.environ["PYTHONBREAKPOINT"] = "pudb.set_trace"
# Loading and Transforming data
train_data_transform = transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomPerspective(),
        transforms.RandomCrop(600),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])
val_data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

# dataset = HAM10000(subset="train", csv_file="HAM10000_metadata.csv", image_folder="HAM10000_images", root_dir="dataset", transform=data_transform)
# train_sampler, val_sampler = get_trainval_samplers(dataset, validation_split=0.2)
# train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, sampler=train_sampler)
# val_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, sampler=val_sampler)
# valset = HAM10000(subset="test", csv_file="HAM10000_metadata.csv", image_folder="HAM10000_images", root_dir="dataset", transform=data_transform)
# val_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)


valset = MVTecAd(subset="val", category="hazelnut", root_dir="dataset/mvtec_anomaly_detection", transform=val_data_transform)
val_loader = DataLoader(valset, batch_size=1, num_workers=4)

num_epochs = 200 
batch_size = 32

trainset = MVTecAd(subset="train", category="hazelnut", root_dir="dataset/mvtec_anomaly_detection", transform=train_data_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=4)

model = Autoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Using", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model = model.to(device)

loss_op = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)
output_folder = os.path.join("output", datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
weights_folder = os.path.join(output_folder, "models")
results_folder = os.path.join(output_folder, "results")
os.makedirs(weights_folder, exist_ok = True)
os.makedirs(results_folder, exist_ok = True)

print("Start training")
for epoch in range(num_epochs):
    model.train(True)
    pbar = tqdm(train_loader)
    pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

    for img in pbar:
        img = img.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = loss_op(output, img)

        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data.cpu().numpy()))
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    model_file = os.path.join(weights_folder, f"model_{epoch}.pth")
    torch.save(state_dict, model_file)
    print(f"Model saved at {model_file}")

    print("Validating")
    model.eval()
    with torch.no_grad():
        for index, img in enumerate(tqdm(val_loader)):
            # img, _ = data
            img = img.to(device)
            output = model(img)
            loss = loss_op(output, img).data.cpu()
            # correct = correct + 1 if loss.item() > 0.5
            image = torch.cat((img, output), 2)
            image = image.cpu()
            result_image = os.path.join(results_folder, f"output_{epoch}_{loss:.4f}.png")
            save_image(image, result_image)
            print(f"Result image saved at {result_image}")

