# Run this after testing completes
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import os

results_dir = './results/maps_pix2pix/val_latest/images/'
ssim_scores = []
psnr_scores = []

for i in range(1, 1043):
    fake_path = os.path.join(results_dir, f'{i}_fake_B.png')
    real_path = os.path.join(results_dir, f'{i}_real_B.png')
    
    if os.path.exists(fake_path) and os.path.exists(real_path):
        fake_img = np.array(Image.open(fake_path))
        real_img = np.array(Image.open(real_path))
        
        ssim_val = ssim(fake_img, real_img, multichannel=True, channel_axis=-1)
        psnr_val = psnr(real_img, fake_img)
        
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)

print(f'SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}')
print(f'PSNR: {np.mean(psnr_scores):.2f} dB ± {np.std(psnr_scores):.2f}')
print(f'Number of test images: {len(ssim_scores)}')
