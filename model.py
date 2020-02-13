import torch
import torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(True),

            nn.Conv2d(16,32,kernel_size=3, stride=2),
            nn.ReLU(True),

            nn.Conv2d(32,8,kernel_size=3),  
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(8,32,kernel_size=3),
            nn.ReLU(True),

            nn.ConvTranspose2d(32,16,kernel_size=3, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(16,3,kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(True))
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x