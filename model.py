import torch
import torch.nn as nn

import torch
import torch.nn as nn
class Bottleneckv2(nn.Module):
    def __init__(self):
        super(Bottleneckv2,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(True),

            nn.Conv2d(16, 8, kernel_size=3, stride=2),
            nn.ReLU(True),

            nn.Conv2d(8, 4, kernel_size=3),  
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(4, 8, kernel_size=3),
            nn.ReLU(True),

            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(True))
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Bottleneckv4(nn.Module):
    def __init__(self):
        super(Bottleneckv4,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 24, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(24),

            nn.Conv2d(24, 32, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3),  
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(32, 32, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 24, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(24),

            nn.ConvTranspose2d(24, 16, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),

            nn.ConvTranspose2d(16, 3, kernel_size=3),
            nn.Sigmoid())
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Bottleneckv5(nn.Module):
    def __init__(self, input_channels=1, latent_dim=100):
        super(Bottleneckv5,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, latent_dim, kernel_size=8, stride=1, padding=0),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(latent_dim, 32, kernel_size=8, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x