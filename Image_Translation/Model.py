import torch
import torch.nn as nn

class downBlock(nn.Module):
    def __init__(self, inChannels, outChannels, batchNorm = True):
        super().__init__()
        
        layers = []
        
        layers.append(nn.Conv2d(
            inChannels,
            outChannels,
            kernel_size = 4,
            stride = 2,
            padding = 1,
            bias = not batchNorm 
            ))    
            
        if batchNorm:
            layers.append(nn.BatchNorm2d(outChannels))
            
        layers.append(nn.LeakyReLU(0.2, inplace = True))
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

        

class upBlock(nn.Module):
    def __init__(self, inChannels, outChannels, dropOut = False):
        super().__init__()
        
        layers = []
        
        layers.append(nn.ConvTranspose2d(
            inChannels, 
            outChannels, 
            kernel_size = 4,
            stride = 2,
            padding = 1
            ))

        layers.append(nn.BatchNorm2d(outChannels))
        
        layers.append(nn.ReLU(inplace=True))
        
        if dropOut:
            layers.append(nn.Dropout2d(0.4))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class GeneratorUnet(nn.Module):
    def __init__(self, inChannels = 3, outChannels = 3):
        super().__init__()
        
        self.down1 = downBlock(inChannels, outChannels = 64, batchNorm = False)
        self.down2 = downBlock(inChannels = 64, outChannels = 128)
        self.down3 = downBlock(inChannels = 128, outChannels = 256)
        self.down4 = downBlock(inChannels = 256, outChannels = 512)
        self.down5 = downBlock(inChannels = 512, outChannels = 512)
        self.down6 = downBlock(inChannels = 512, outChannels = 512)
        self.down7 = downBlock(inChannels = 512, outChannels = 512)
        self.down8 = downBlock(inChannels = 512, outChannels = 512, batchNorm = False)

        self.up1 = upBlock(inChannels = 512, outChannels = 512, dropOut = True)
        self.up2 = upBlock(inChannels = 1024, outChannels = 512, dropOut = True)
        self.up3 = upBlock(inChannels = 1024, outChannels = 512, dropOut = True)
        self.up4 = upBlock(inChannels = 1024, outChannels = 512)
        self.up5 = upBlock(inChannels = 1024, outChannels = 256)
        self.up6 = upBlock(inChannels = 256 + 256, outChannels = 128)
        self.up7 = upBlock(inChannels = 128 + 128, outChannels = 64)
        self.up8 = nn.ConvTranspose2d(in_channels = 64 + 64, out_channels = outChannels,
                                      kernel_size = 4, stride = 2, padding = 1)
        self.activeF = nn.Tanh()
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        u1 = self.up1(d8)
        u2 = self.up2(torch.concat([u1,d7], dim = 1))
        u3 = self.up3(torch.concat([u2,d6], dim = 1))
        u4 = self.up4(torch.concat([u3,d5], dim = 1))
        u5 = self.up5(torch.concat([u4,d4], dim = 1))
        u6 = self.up6(torch.concat([u5,d3], dim = 1))
        u7 = self.up7(torch.concat([u6,d2], dim = 1))
        output = self.activeF(self.up8(torch.concat([u7,d1], dim = 1)))
        
        return output
    
    
class discriminatorPatchGAN(nn.Module):
    def __init__(self, inChannel = 6):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels = inChannel, out_channels = 64, kernel_size = 4 
                      ,stride = 2, padding = 1),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4 
                      ,stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4 
                      ,stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4 
                      ,stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            
            nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4 
                      ,stride = 1, padding = 1)
            )
        
    def forward(self, x, y):
        return self.discriminator(torch.cat([x, y], dim = 1))
        
        