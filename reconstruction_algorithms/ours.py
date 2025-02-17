from functools import cache
import itertools
import json
import random
import torch as pt
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image, ImageReadMode
import numpy as np
import time
import torchvision
from skimage.filters import threshold_multiotsu
from utils import filter_sinogram,FBPRadon
from regularization import (create_autoencoder_regularization,
                            binary_regularization,
                            total_variation_regularization,
                            tikhonov_regularization,
                            )
from pytorch_models import HTCModel
from ReconstructorBase import NoModel, HTCModelReconstructor

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

class LPLoss(nn.Module):
    def __init__(self, p=1.0):
        """ Loss that computes the Lp norm of the difference between two images.
        """
        super(LPLoss, self).__init__()
        self.p = p
        
    def forward(self, X, Y):
        # calculate the mean of the Lp norm of the difference between X and Y
        return pt.mean(pt.abs(X - Y)**self.p * (1/self.p))

def create_regularization(use_tv_reg = True,
                          use_bin_reg = False,
                          use_tik_reg = False,
                          use_autoencoder_reg = False,
                          autoencoder_path="",
                          autoencoder_patch_size=40,
                          autoencoder_latent_vars=10,
                          autoencoder_reconstruction_stride=5,
                          autoencoder_batch_size=128,
    ):
    regularizations = []
    if use_tv_reg:
        use_tv_reg = 1 if isinstance(use_tv_reg,bool) else use_tv_reg
        regularizations.append(lambda x : use_tv_reg * total_variation_regularization(x,normalize=True))
    if use_bin_reg:
        use_bin_reg = 1 if isinstance(use_bin_reg,bool) else use_bin_reg
        regularizations.append(lambda x : use_bin_reg * binary_regularization(x))
    if use_tik_reg:
        use_tik_reg = 1 if isinstance(use_tik_reg,bool) else use_tik_reg
        regularizations.append(lambda x : use_tik_reg * tikhonov_regularization(x,normalize=True))
    if use_autoencoder_reg:
        use_autoencoder_reg = 1 if isinstance(use_autoencoder_reg,bool) else use_autoencoder_reg
        regu = create_autoencoder_regularization(autoencoder_path,
                                                autoencoder_patch_size,
                                                autoencoder_latent_vars,
                                                autoencoder_reconstruction_stride,
                                                autoencoder_batch_size,
                                                )
        reg = lambda x : use_autoencoder_reg * regu(x)
        regularizations.append(reg)
        
    if not regularizations:
        return lambda x: pt.tensor(0.0, device=x.device)
    return lambda x: sum([r(x) for r in regularizations])



def reconstruct(sinogram, angles,kwargs_path="best_params.json"):
    pt.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    pt.set_default_device('cuda')
    with open(kwargs_path, 'r') as f:
        params = json.load(f)
        params = params["params"]
    
    sinogram = pt.tensor(sinogram, dtype=pt.float32, device='cuda')
    
    mask = load_solid_mask(params.get("mask_buffer",1))
    if params["trim_sinogram"]:
        num_extra_cols = sinogram.shape[1] - 512
        if num_extra_cols != 0:
            rm_from_left = num_extra_cols // 2 - 1
            rm_from_right = num_extra_cols - rm_from_left
            sinogram = sinogram[:, rm_from_left:sinogram.shape[1] - rm_from_right]
    
    if params["use_no_model"]:
        model = NoModel(
            proj_dim=sinogram.shape[1],
            angles=np.deg2rad(angles),
            a=params["filter_sinogram_of_predicted_image_with_a"],
            image_mask=mask,
            edge_pad_size=params["edge_pad_size"]
        )
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], amsgrad=True)
    else:
        model = HTCModelReconstructor(
            proj_dim=sinogram.shape[1],
            angles=np.deg2rad(angles),
            a=params["filter_sinogram_of_predicted_image_with_a"],
            image_mask=mask,
            edge_pad_size=params["edge_pad_size"]
        )
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], amsgrad=True)
    model.to('cuda')
    
    raw_sinogram = sinogram
    if params["filter_raw_sinogram_with_a"] != 0:
        filtered_sinogram = filter_sinogram(raw_sinogram, params["filter_raw_sinogram_with_a"], device="cuda")
    else:
        filtered_sinogram = raw_sinogram
        filtered_sinogram = (filtered_sinogram - pt.mean(filtered_sinogram)) / pt.std(filtered_sinogram)
    
    criterion = LPLoss(p=params["p_loss"])
    
    regularization_ = create_regularization(
        use_tv_reg=params["use_tv_reg"],
        use_bin_reg=params["use_bin_reg"],
        use_tik_reg=params["use_tik_reg"],
        use_autoencoder_reg=params["use_autoencoder_reg"],
        autoencoder_path=params["autoencoder_path"],
        autoencoder_patch_size=params["autoencoder_patch_size"],
        autoencoder_latent_vars=params["autoencoder_latent_vars"],
        autoencoder_reconstruction_stride=params["autoencoder_reconstruction_stride"],
        autoencoder_batch_size=params["autoencoder_batch_size"],
    )
    
    iteration_number = 0
    start_time = time.time()
    while iteration_number < params["num_iters"]:

        y_hat, s_hat = model(raw_sinogram)
        y_hat = y_hat.reshape(mask.shape)
        s_hat = s_hat.reshape((len(angles), 512))

        criterion_loss = criterion(filtered_sinogram, s_hat)
        regularization_loss = regularization_(y_hat)
        loss = criterion_loss + regularization_loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        y_hat_np = y_hat.cpu().detach().numpy()
        if params["use_otsu"]:
            thresholds = threshold_multiotsu(y_hat_np, 2)
            y_hat_np = np.digitize(y_hat_np, bins=thresholds).astype(np.uint8)
        
        y_hat_np = np.clip(y_hat_np,0,1)
        y_hat_rounded_np = np.round(y_hat_np)

        iteration_number += 1

        if iteration_number % 50 == 0:
            s = f"Iteration: {iteration_number}, Loss: {loss.item()}, Criterion loss: {criterion_loss.item()}, Regularization loss: {regularization_loss.item()}"
            print(s)
    return y_hat_rounded_np



    
    
    