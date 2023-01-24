import torch
import cv2
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

def transpose_and_prepare(data):
    N, C, H, W = data.shape
    data = (data * 0.5) + 0.5
    data = (data.cpu().detach().numpy())
    
    data = (data * 255).astype(np.uint8)
    data = np.transpose(data, (0,2,3,1))
    new_data = np.empty((len(data), H, W, 3)).astype(np.uint8)
    for i in range(len(new_data)):
        new_data[i] = cv2.cvtColor(data[i], cv2.COLOR_GRAY2RGB)
    new_data = np.transpose(new_data, (0,3,1,2))
    return torch.from_numpy(new_data)
    
    
    
class EvaluateGAN():
    def __init__(self, device, features:int = 64, batch_size:int = 128):
        self.device = device
        self.fid = FrechetInceptionDistance(feature=64, reset_real_features=False).to(self.device)
        self.kid = KernelInceptionDistance(feature=64, subset_size=batch_size, reset_real_features=False).to(self.device)

        self.current_fid = np.inf
        self.previous_fid = np.inf
        
        self.current_kid = (np.inf, np.inf)
        self.previous_kid = (np.inf, np.inf)
        
    def calculate_statistics_for_real(self, dataloader):
        for data in tqdm(dataloader, total=len(dataloader)):
            images = transpose_and_prepare(data)
            images = images.to(self.device)
            self.fid.update(images, real=True)
            self.kid.update(images, real=True)
            
            
    def calculate_statistics_for_fake(self, generator, num_samples, device, nz=100):
        for _ in tqdm(range(int(num_samples / 128))):
            fixed_noise = torch.randn(128, nz, 1, 1, device=device)
            with torch.no_grad():
                images = generator(fixed_noise)
            
            images = transpose_and_prepare(images)
            images = images.to(device)
            
            self.fid.update(images, real=False)
            self.kid.update(images, real=False)
        
    def map_tensors(self, data):
        return tuple(map(lambda x: float(x), data)) 
        
    def print_statistics(self):
        print('current FID / previous FID [(%.4f/%.4f), ratio: (%.4f)]'
             % (self.current_fid, self.previous_fid, self.current_fid / self.previous_fid))
        print(f'current KID / previous KID [{self.map_tensors(self.current_kid)}/{self.map_tensors(self.previous_kid)}]')
        
        
    def evaluate(self, generator, num_samples, device):
        print()
        self.calculate_statistics_for_fake(generator, num_samples, device)
        self.current_fid = float(self.fid.compute().cpu().detach().numpy())
        self.current_kid = self.kid.compute()
        self.print_statistics()
        self.previous_fid = self.current_fid
        self.previous_kid = self.current_kid
        self.fid.reset()
        self.kid.reset()
        print()
        return self.current_fid, self.map_tensors(self.current_kid)
            
    
    