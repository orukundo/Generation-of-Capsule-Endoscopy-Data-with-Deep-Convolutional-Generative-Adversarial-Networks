# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 18:10:45 2025

@author: oruku
"""

import torch
import torch.nn as nn
import torchvision.utils as vutils
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
from PIL import Image, ImageEnhance

# Generator
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            nn.Conv2d(nc, nc, kernel_size=33, stride=1, padding=0)
        )

    def forward(self, input):
        return self.main(input)

# Load generator
def load_generator(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    netG = Generator(nz=config['nz'], ngf=config['ngf'], nc=config['nc']).to(device)
    netG.load_state_dict(checkpoint['state_dict'])
    netG.eval()
    return netG

# Generate synthetic images
def run_gui():
    root = tk.Tk()
    root.withdraw()

    # Select model
    gen_path = filedialog.askopenfilename(title="Select generator.pth file")
    if not gen_path:
        print("No generator selected.")
        return

    # Select output folder
    out_folder = filedialog.askdirectory(title="Select output folder")
    if not out_folder:
        print("No output folder selected.")
        return

    num_images = simpledialog.askinteger("Input", "How many images to generate?", minvalue=1, maxvalue=9999)
    if not num_images:
        print("No number entered.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG = load_generator(gen_path, device)

    nz = 100
    with torch.no_grad():
        noise = torch.randn(num_images, nz, 1, 1, device=device)
        fake_images = netG(noise).detach().cpu()
        
        for i, img in enumerate(fake_images):
            temp_path = os.path.join(out_folder, f"synthetic_{i+1}.png")
            enhanced_path = os.path.join(out_folder, f"enhanced_{i+1}.png")
            vutils.save_image(img, temp_path, normalize=True)
            pil_img = Image.open(temp_path)
            pil_img = ImageEnhance.Contrast(pil_img).enhance(1.6)
            pil_img.save(enhanced_path)
            os.remove(temp_path)
        
    print(f"\nSaved {num_images} synthetic images to:\n{out_folder}")

if __name__ == '__main__':
    run_gui()
