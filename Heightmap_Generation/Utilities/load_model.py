import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import os
import matplotlib.animation as animation
from tqdm import tqdm
import random

random.seed(1)

SCALE_FACTORS = {'Death_Valley': (-83.09072972922985, 575.7622747857106),
                 'Mt_Rainer': (588.9312975792448, 3121.475279966315),
                 'River_Basin': (2193.9686848516926, 3433.3536631618485),
                 'San_Gabriel': (485.01180030981095, 1878.399252241416),
                 'Laytonville': (147.52433426903116, 980.0772495763101),
                 'Post_Earthquake': (665.8615940945722, 920.7708323530042)}

class Generator_256(nn.Module):
    def __init__(self):
        nz = 100
        nc = 1
        ngf = 64
        super(Generator_256, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution

            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nz, ngf * 16, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 16, ngf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1, bias=False),

            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf * 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 1, int(ngf * 0.5), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(ngf * 0.5)),
            nn.ReLU(True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(int(ngf * 0.5), int(ngf * 0.25), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(ngf * 0.25)),
            nn.ReLU(True),
            
            # state size. (ngf) x 32 x 32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(int(ngf * 0.25), nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)




class GenerationModels:
    def __init__(self, name, path=None, device='cpu'):
        self.name = name
        self.path = path
        self.device = device
        self.generator = self._load_model()
        self.generator.eval()



    def _load_model(self):

        generator = Generator_256()
        
        generator = generator.to(self.device)

        if self.path:
            checkpoint = torch.load(self.path, map_location=torch.device(device=self.device))
            generator.load_state_dict(checkpoint['generator'])
        
        return generator

    def create_heightmaps(self, save_path, n_images, base_name_image='generated_image_', save=True, random_seed=-1):
        if random_seed != -1:
            _ = torch.manual_seed(random_seed)
        save_path = Path(save_path)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        image_name = base_name_image + "{:2d}.png"
        fixed_noise = torch.randn(n_images, 100, 1, 1, device=self.device)
        generated_images = self.generator(fixed_noise).cpu().detach().numpy()[:, 0, :, :]
        generated_images_scaled = self.scale_to_real_heights(generated_images)
        

        if save:
            for i in range(n_images):
                current_name = image_name.format(i+1) 
                plt.imsave(save_path / current_name, generated_images[i], cmap='gray')
        else:
            return generated_images, generated_images_scaled
        return None, None

    

    def create_heightmaps_n_plets(self, save_path, n_images, path_real, plet=3, base_name_image='generated_image_plet_', save=True, random_seed=-1):
        if random_seed != -1:
            _ = torch.manual_seed(random_seed)
        save_path = Path(save_path)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        image_name = base_name_image + "{:2d}.png"
        fixed_noise = torch.randn(n_images*plet, 100, 1, 1, device=self.device)
        generated_images = self.generator(fixed_noise).cpu().detach().numpy()[:, 0, :, :]
        generated_images_scaled = self.scale_to_real_heights(generated_images)

        real_images = np.load(path_real)[:1500]
        print('loaded')
        

        if save:
            for i in range(n_images):
                current_name = image_name.format(i+1)
                nrows = 2
                fig, axs = plt.subplots(nrows=nrows, ncols=plet, figsize=(5*plet, 5*nrows))
                for j in range(plet):
                    axs[0, j].imshow(generated_images[i*3+j], cmap='gray')
                    random_number = random.randint(0, len(real_images)-1)
                    axs[1, j].imshow(real_images[random_number], cmap='gray')
                    plt.savefig(save_path / current_name)
                plt.imsave(save_path / current_name, generated_images[i], cmap='gray')
        else:
            return generated_images, generated_images_scaled
        return None, None

    

    def scale_to_real_heights(self, heightmap):
        heightmap = (heightmap * 0.5) + 0.5
        heightmap = heightmap * (SCALE_FACTORS[self.name][1] - SCALE_FACTORS[self.name][0]) + SCALE_FACTORS[self.name][0]
        return heightmap
    
    def morph_heightmaps(self, save_path, n_images, n_gifs=1, normal_range=3, base_name_gif='generated_gif_', save=False, random_seed=-1):
        if random_seed != -1:
            _ = torch.manual_seed(random_seed)

        save_path = Path(save_path)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        gif_name = base_name_gif + "{:2d}.gif"

        
        for j in tqdm(range(n_gifs), total=n_gifs):
            random_noise = torch.randn(1, 100, 1, 1, device=self.device)

            all_images = np.empty((n_images, 100, 1, 1))
            lin_spaced_values = np.linspace(-normal_range, normal_range, n_images)
            all_images[0] = random_noise
            for i in range(len(all_images)):
                all_images[i] = random_noise + lin_spaced_values[i]
            
            all_images = torch.from_numpy(all_images.astype(np.float32)).to(device=self.device)
                

            generated_images = self.generator(all_images).cpu().detach().numpy()[:, 0, :, :]
            generated_images_scaled = self.scale_to_real_heights(generated_images)
            
            
            if save:
                name = str(save_path).split('/')[-1]
                anim = self.create_animation(name, generated_images)
                writer = animation.FFMpegWriter(fps=10)
                anim.save(save_path / gif_name.format(j), writer=writer)
                
            else:
                return generated_images, generated_images_scaled
        return None, None

    def create_animation(self, name, images):
        fig = plt.figure(figsize=(6,6))
        fig.suptitle(name)
        plt.axis("off")
        ims = [[plt.imshow(i, cmap='gray', animated=True)] for i in images]
        ani = animation.ArtistAnimation(fig, ims, interval=250, repeat_delay=1000, blit=True)

        return ani


