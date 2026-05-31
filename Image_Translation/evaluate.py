import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

class evluateModel:
    def __init__(self, generator, dataLoader, device):
        self.G = generator
        self.data = dataLoader
        self.device = device
    
        self.l1Loss = nn.L1Loss(reduction = 'mean')
        self.l2Loss = nn.MSELoss(reduction = 'mean')
        self.psnr = peak_signal_noise_ratio
        self.ssim = structural_similarity
        
    def denormalize(self, x):
        return (x + 1) / 2
        
    def testEvaluatel1l2(self):
        self.G.eval()
        
        l1LossEval = 0
        l2LossEval = 0
        batchNum = 0
        
        with torch.no_grad():
            for X, Y in self.data:
                X = X.to(self.device)
                Y = Y.to(self.device)
                
                generate = self.G(X)
                
                l1LossEval += self.l1Loss(Y, generate).item()
                l2LossEval += self.l2Loss(Y, generate).item()
                batchNum += 1
                
        averagel1Loss = l1LossEval / batchNum
        averagel2Loss = l2LossEval / batchNum
        
        print(f"average l1Loss is {averagel1Loss}")
        print(f"average l2Loss is {averagel2Loss}")

        return averagel1Loss, averagel2Loss
    
    def testEvaluatePSNR_SSIM(self):
        self.G.eval()
        
        psnr = []
        ssim = []
        
        with torch.no_grad():
            for X, Y in self.data:
                X = X.to(self.device)
                Y = Y.to(self.device)
                
                generate = self.G(X)
                
                Y = self.denormalize(Y).cpu().numpy()
                generate = self.denormalize(generate).cpu().numpy()
                
                for i in range(generate.shape[0]):
                    psnri = self.psnr(Y[i], generate[i], data_range = 1.0)
                    
                    ssimi = self.ssim(Y[i].transpose(1,2,0),
                                     generate[i].transpose(1,2,0),
                                     data_range = 1.0,
                                     channel_axis = 2)
                    
                    psnr.append(psnri)
                    ssim.append(ssimi)
                    
        print(f"average psnr is {np.mean(psnr)}")
        print(f"average ssim is {np.mean(ssim)}")        
        
        return psnr, ssim      
                
    def visualize(self, samples):
        self.G.eval()
        dataset = self.data.dataset

        _, ax = plt.subplots(ncols = 3, nrows = samples, figsize = (9, 3*samples))
        
        idx = np.random.randint(0, len(self.data), size = samples)
        
        for i,j in zip(idx,range(len(ax))):
            X, Y = dataset[i]
            X = X.to(self.device)  
            Y = Y.to(self.device)
            
            with torch.no_grad():
                fake = self.G(X.unsqueeze(0))
            
            X = self.denormalize(X).cpu().numpy()
            Y = self.denormalize(Y).cpu().numpy()
            fake = self.denormalize(fake[0]).cpu().numpy()
            print(fake.shape)
            ax[j,0].imshow(X.transpose(1, 2, 0))
            ax[j,0].set_title("Input Image")

            ax[j,1].imshow(fake.transpose(1, 2, 0))
            ax[j,1].set_title("Generated Image")

            ax[j,2].imshow(Y.transpose(1, 2, 0))
            ax[j,2].set_title("Labeled Image")

        plt.tight_layout()
        plt.show()
        
                
        
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    