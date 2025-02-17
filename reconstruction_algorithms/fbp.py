from functools import cache
import numpy as np
from skimage.transform import iradon
from torchvision.io import read_image, ImageReadMode
import torch as pt
import torchvision

@cache
def load_solid_mask(buffer = 1):
    img = read_image("./solid_disk.png", ImageReadMode.GRAY)
    img = img / 255
    img = img.squeeze()
    #return img.to("cuda")
    buffered_mask = pt.zeros_like(img)
    
    # Use dilation to add buffer around the disk
    kernel = pt.ones((2 * buffer + 1, 2 * buffer + 1), device=img.device)
    buffered_mask = pt.nn.functional.conv2d(img.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=buffer)
    buffered_mask = pt.clamp(buffered_mask, 0, 1).squeeze()
    buffered_mask = buffered_mask.to("cuda")
    return buffered_mask

def reconstruct(sinogram, angles):
    """
    Reconstructs an image from its sinogram using filtered backprojection.

    Args:
        sinogram (numpy.ndarray): The sinogram of the image.
        angles (numpy.ndarray): The angles at which the sinogram was taken, in degrees.

    Returns:
        numpy.ndarray: The reconstructed image.
    """
    mask = load_solid_mask().cpu().numpy()
    num_extra_cols = sinogram.shape[1] - 512
    if num_extra_cols != 0:
        rm_from_left = num_extra_cols // 2 - 1
        rm_from_right = num_extra_cols - rm_from_left
        sinogram = sinogram[:, rm_from_left:sinogram.shape[1] - rm_from_right]
        print(f"Trimmed sinogram to shape: {sinogram.shape}")
    reconstruction = iradon(sinogram.cpu().numpy().T, theta=angles, filter_name='ramp')
    reconstruction = mask * reconstruction
    # Normalize the reconstruction to [0, 1].
    #reconstruction = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())
    # Binarize
    reconstruction = (reconstruction > 0.5).astype(np.float32)
    return reconstruction
