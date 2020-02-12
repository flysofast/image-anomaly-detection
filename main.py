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
from tensorboardX import SummaryWriter
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


valset = MVTecAd(subset="val", category="hazelnut", root_dir="dataset/mvtec_anomaly_detection", transform=val_data_transform)
val_loader = DataLoader(valset, batch_size=1, num_workers=4)

num_epochs = 200 
batch_size = 32
save_interval = 10
trainset = MVTecAd(subset="train", category="hazelnut", root_dir="dataset/mvtec_anomaly_detection", transform=train_data_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=4)

model = Autoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Using", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model = model.to(device)

loss_op = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
exp_name = f'AE_{datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")}'
output_folder = os.path.join("output", exp_name)
weights_folder = os.path.join(output_folder, "models")
results_folder = os.path.join(output_folder, "results")
os.makedirs(weights_folder, exist_ok = True)
os.makedirs(results_folder, exist_ok = True)

writer = SummaryWriter(log_dir=f"log/{exp_name}")

print("Start training")
min_loss = 1e10
for epoch in range(num_epochs):
    model.train(True)
    pbar = tqdm(train_loader)
    pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
    epoch_loss = 0
    for i, img in enumerate(pbar):
        img = img.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = loss_op(output, img)

        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        epoch_loss += loss_val
        writer.add_scalar("MSE/train", loss_val, epoch * len(train_loader) + i)


    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, epoch_loss/len(train_loader)))

    if epoch_loss < min_loss or epoch % save_interval == 0:
        
        if isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        model_file = os.path.join(weights_folder, f"model_{epoch}.pth")
        torch.save(state_dict, model_file)
        print(f"Model saved at {model_file}")

    print("Validating...")
    model.eval()
    with torch.no_grad():
        for index, img in enumerate(tqdm(val_loader)):
            # img, _ = data
            img = img.to(device)
            output = model(img)
            loss = loss_op(output, img).data.cpu()
            
            if epoch_loss < min_loss or epoch % save_interval == 0:
                diff = torch.abs(img - output)
                
                # Repeat one channeled mean values along channel axis to the same shape of an image
                diff_avg = torch.mean(diff, dim=1).unsqueeze(1).expand(-1, 3, -1, -1)
                
                image = torch.cat((img, output, diff, diff_avg), 0)
                image = image.cpu()
                result_image = os.path.join(results_folder, f"output_{epoch}_{loss:.4f}.png")

                # create grid of images
                img_grid = torchvision.utils.make_grid(image)
                torchvision.utils.save_image(image, result_image, nrow=image.shape[0])
                print(f"Result images saved at {results_folder}")

            # write to tensorboard
            writer.add_scalar("MSE/val", loss.item(), global_step=epoch)
            writer.add_image('Val images', img_grid, global_step=epoch)
    
    if epoch_loss < min_loss:
        min_loss = epoch_loss

