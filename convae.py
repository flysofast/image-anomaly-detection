import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import os

# Loading and Transforming data
transform = transforms.Compose([
                transforms.ToTensor(),  
                transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))
            ])
trainTransform  = tv.transforms.Compose([
                    tv.transforms.ToTensor(), 
                    # tv.transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))
                ])


# Writing our model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(16,6,kernel_size=3),
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(6,16,kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,3,kernel_size=3),
            nn.ReLU(True))
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#defining some params
num_epochs = 200 
batch_size = 128

trainset = tv.datasets.CIFAR10(root='./data',  train=True,download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testset = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

model = Autoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Using", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model = model.to(device)

distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)
output_folder = "output"
weights_folder = os.path.join(output_folder, "weights")
image_folder = os.path.join(output_folder, "image")

os.makedirs(weights_folder, exist_ok = True)
for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.to(device)
        optimizer.zero_grad()
        
        # ===================forward=====================
        output = model(img)
        loss = distance(output, img)
        # ===================backward====================
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data.cpu().numpy()))
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = mode.state_dict()
    torch.save(state_dict, os.path.join(weights_folder, f"model_{epoch}.pth"))
    for data in testloader:
        img, _ = data
        img=img.to(device)
        output = model(img)
        output = output.detach()
        image = torch.cat((img, output), 2)
        image = image.cpu()
        save_image(image,os.path.join(image_folder, f"output_{epoch}.png") )

