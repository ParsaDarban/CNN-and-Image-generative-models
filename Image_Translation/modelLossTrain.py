import torch
import torch.nn as nn

class training:
    def __init__(self, generator, discriminator, 
                 optimG, optimD, device, lambdaL1 = 100):
        self.G = generator
        self.D = discriminator
        self.device = device
        self.optimG = optimG
        self.optimD = optimD
        self.lambdaL1 = lambdaL1
        
        self.ganLoss = nn.BCEWithLogitsLoss()
        self.l1Loss = nn.L1Loss()
        
    def discriminatorLoss(self, realX, realY, fakeY):
        real = self.D(realX, realY)
        realLoss = self.ganLoss(real, torch.ones_like(real))
        
        fake = self.D(realX, fakeY.detach())
        fakeLoss = self.ganLoss(fake, torch.zeros_like(fake))
        
        lossD = 0.5 * (realLoss + fakeLoss)
        return lossD
    
    def generatorLoss(self, realX, realY, fakeY):
        fakeD = self.D(realX, fakeY)
        fakeLoss = self.ganLoss(fakeD, torch.ones_like(fakeD))
        
        generateLoss = self.l1Loss(realY, fakeY)
        
        lossG = self.lambdaL1 * generateLoss + fakeLoss
        
        return lossG, fakeLoss, generateLoss
    
    def training(self, realX, realY):
        realX = realX.to(self.device)
        realY = realY.to(self.device)
        
        fakeY = self.G(realX)
        self.optimD.zero_grad()
        lossD = self.discriminatorLoss(realX, realY, fakeY)
        lossD.backward()
        self.optimD.step()
        
        self.optimG.zero_grad()
        lossG, generateLoss, l1loss = self.generatorLoss(realX, realY, fakeY)
        lossG.backward()
        self.optimG.step()        
        
        return{
            "loss_D": lossD.item(),
            "loss_G": lossG.item(),
            "loss_G_GAN": generateLoss.item(),
            "loss_G_L1": l1loss.item(),
        }
        
        
        
        
        