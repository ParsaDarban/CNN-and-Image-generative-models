import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os

class AerialDataset(Dataset):
    def __init__(self, folder, imageSize = 256):
        self.folder = folder
        self.imageSize = imageSize
        self.path = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png"))
            ])
        
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        
        imgPath = self.path[idx]
        img = cv.imread(imgPath)
        
        if img is None:
            raise ValueError (f"Error in reading {imgPath}")
        
        # imgRGB can be change to img too
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        imgResize = cv.resize(imgRGB, 
                              (self.imageSize * 2, self.imageSize),
                              interpolation=cv.INTER_AREA)
        
        
        X = imgResize[:,:self.imageSize,:]
        y = imgResize[:,self.imageSize: ,:]
        
        X = (X / 127.5) - 1
        y = (y / 127.5) - 1
                
        X = torch.from_numpy(X).permute(2, 0, 1).float()
        y = torch.from_numpy(y).permute(2, 0, 1).float()
        return X, y
        


#plt.imshow(trainData[0])        
# DataLoader(
#     )
        
        
        
        
