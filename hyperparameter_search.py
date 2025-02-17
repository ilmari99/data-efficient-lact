from functools import cache
import itertools
import json
import os
import random
import torch as pt
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image, ImageReadMode
import numpy as np
import time
from skimage.filters import threshold_multiotsu
from tqdm import tqdm
from utils import calc_dice_similarity, calc_phi_coefficient, filter_sinogram
from regularization import (create_autoencoder_regularization,
                            binary_regularization,
                            total_variation_regularization,
                            tikhonov_regularization,
                            )
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

def get_htc_scan(level = 1, sample = "a"):
    base_path = "./htc2022_data/"
    # htc2022_01a_recon_fbp_seg.png
    htc_file = f"htc2022_0{level}{sample}_recon_fbp_seg.png"
    sinogram_file = f"htc2022_0{level}{sample}_limited_sinogram.csv"
    angle_file = f"htc2022_0{level}{sample}_angles.csv"
    angle_file = os.path.join(base_path, angle_file)
    
    # Load angles
    angles = np.loadtxt(angle_file,dtype=np.str_, delimiter=",")
    angles = np.array([float(angle) for angle in angles])

    # Load the img
    img = read_image(os.path.join(base_path, htc_file))
    #img = pt.tensor(img, dtype=pt.float32, device='cuda', requires_grad=False)
    print(f"Sample {level}{sample} rotating to {angles[0]}")
    img = img.squeeze()
    y = img / 255
    
    outer_mask = load_solid_mask()
    sinogram = np.loadtxt(os.path.join(base_path, sinogram_file), delimiter=',')
    sinogram = pt.tensor(sinogram) * 255
    
    if level == 8:
        sample_to_degrees = {
            "a": 30,
            "b": 30,
            "c": 30,
            "d": 30,
        }
        # Only take a part of the sinogram
        angles = np.arange(0,sample_to_degrees[sample] + 0.5,0.5)
        sinogram = sinogram[0:len(angles),:]
    return y, outer_mask, sinogram, angles

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
    autoencoder_path = "autoencoders/" + autoencoder_path if autoencoder_path else ""
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
                          

def folder_name_from_params(base_name = "Benchmark", **kwargs):
    # Sort alphabetically
    kwargs = dict(sorted(kwargs.items()))
    for k,v in kwargs.items():
        base_name += "_" + str(k) + "=" + str(v)
    return base_name


def reconstruct(sinogram, angles,kwargs):
    params = kwargs
    sinogram = pt.tensor(sinogram)
    pt.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    
    if params["filter_sinogram_of_predicted_image_with_a"] == -1:
        params["filter_sinogram_of_predicted_image_with_a"] = params["filter_raw_sinogram_with_a"]
    
    mask = load_solid_mask(params["mask_buffer"])
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
            edge_pad_size=params["edge_pad_size"],
            shift_mask=params["shift_mask"]
        )
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], amsgrad=True)
    else:
        model = HTCModelReconstructor(
            proj_dim=sinogram.shape[1],
            angles=np.deg2rad(angles),
            a=params["filter_sinogram_of_predicted_image_with_a"],
            image_mask=mask,
            edge_pad_size=params["edge_pad_size"],
            shift_mask=params["shift_mask"]
        )
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], amsgrad=True)
    model.to('cuda')
    
    raw_sinogram = sinogram
    if params["filter_raw_sinogram_with_a"] != 0:
        filtered_sinogram = filter_sinogram(raw_sinogram, params["filter_raw_sinogram_with_a"])
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


def test_on_sample(level, sample, reconstruction_kwargs):
    # Load sinogram and angles
    true_image, _, sinogram, angles = get_htc_scan(level,sample)
    true_image = true_image.detach().cpu().numpy()
    # Reconstruct the image
    start_time = time.time()
    reconstructed_image = reconstruct(sinogram, angles, reconstruction_kwargs)
    end_time = time.time()
    print(f"Reconstruction took {end_time - start_time} seconds")
    # Calculate the error
    phi = float(calc_phi_coefficient(true_image, reconstructed_image))
    dice = float(calc_dice_similarity(true_image, reconstructed_image))
    return {"mcc":phi, "dice":dice, "time":end_time - start_time}

def test_on_level(level, reconstruction_kwargs):
    samples = ["a","b","c"] if level < 8 else ["a","b","c","d"]
    results = {}
    for sample in samples:
        results[sample] = test_on_sample(level,sample,reconstruction_kwargs)
    results["total_mcc"] = sum([results[sample]["mcc"] for sample in samples])
    results["total_dice"] = sum([results[sample]["dice"] for sample in samples])
    results["total_time"] = sum([results[sample]["time"] for sample in samples])
    return results

def test_on_levels(levels, reconstruction_kwargs):
    results = {}
    for level in levels:
        results[level] = test_on_level(level,reconstruction_kwargs)
    return results

def test_parameters(params, levels=[8], overwrite=False, folder_name="hyperparameters"):
    use_no_model = params["use_no_model"]
    p_loss = params["p_loss"]
    filter_raw_sinogram_with_a = params["filter_raw_sinogram_with_a"]
    num_iters = params["num_iters"]
    use_otsu = params["use_otsu"]
    use_tv_reg = params["use_tv_reg"]
    use_bin_reg = params["use_bin_reg"]
    use_tik_reg = params["use_tik_reg"]
    use_autoencoder_reg = params["use_autoencoder_reg"]
    autoencoder_patch_size = params["autoencoder_patch_size"]
    autoencoder_latent_vars = params["autoencoder_latent_vars"]
    autoencoder_reconstruction_stride = params["autoencoder_reconstruction_stride"]
    file_prefix = params["file_prefix"]

    kwargs_for_naming = {
        "Model": not use_no_model,
        "P": p_loss,
        "Filt": filter_raw_sinogram_with_a,
        "Iters": num_iters,
        "use_otsu": use_otsu,
    }

    if use_tv_reg:
        kwargs_for_naming["TV"] = str(use_tv_reg)
    if use_bin_reg:
        kwargs_for_naming["BinReg"] = str(use_bin_reg)
    if use_tik_reg:
        kwargs_for_naming["TK"] = str(use_tik_reg)
    if use_autoencoder_reg:
        kwargs_for_naming["AutoEnc"] = f"Patch{autoencoder_patch_size}LV{autoencoder_latent_vars}"
        kwargs_for_naming["AutoEnc"] += f"Stride{autoencoder_reconstruction_stride}"
        kwargs_for_naming["Coeff"] = use_autoencoder_reg

    filename = folder_name_from_params(base_name=file_prefix, **kwargs_for_naming)
    filename = os.path.join(folder_name,filename) + "_results.json"
    if os.path.exists(filename) and not overwrite:
        print(f"Results file {filename}_results.json already exists. Not overwriting.")
        return
    
    results = test_on_levels(levels, params)
    # Add the parameters to the results
    results["params"] = params
    os.makedirs(folder_name, exist_ok=True)
    # Write the results to a file
    print(results)
    with open(filename, "w") as f:
        json.dump(results, f)
    return results

def run_from_file(param_path, levels=[8], overwrite=False):
    with open(param_path, "r") as f:
        params = json.load(f)
    return test_parameters(params, levels, overwrite)


def set_default_params(params, overwrite=False):
    default_params = {
        "edge_pad_size": 0,
        "file_prefix": "HPTuning",
        "trim_sinogram": True,
        "use_otsu": False,
        "use_bin_reg": 0,
        "use_tik_reg": 0,
        "autoencoder_path": "",
        "p_loss": 1.0,
        "filter_sinogram_of_predicted_image_with_a": -1,
        "autoencoder_reconstruction_stride": 0,
        "autoencoder_latent_vars": 0,
        "autoencoder_batch_size": 128,
        "autoencoder_patch_size": 0,
        "shift_mask": True,
        "mask_buffer": 1,
    }
    if "iteration_number" in params:
        params["num_iters"] = params["iteration_number"]
        del params["iteration_number"]
        
    for k,v in default_params.items():
        if k not in params or overwrite:
            params[k] = v
            
            
    if params["autoencoder_latent_vars"] == 0:
        params["autoencoder_latent_vars"] = params["autoencoder_patch_size"] // 4
    if not params["autoencoder_path"]:
        params["autoencoder_path"] = f"patch_autoencoder_P{params['autoencoder_patch_size']}_D{params['autoencoder_latent_vars']}.pth"
    if params["autoencoder_reconstruction_stride"] == 0:
        params["autoencoder_reconstruction_stride"] = params["autoencoder_patch_size"]
    return params
    

if __name__ == "__main__":
    pt.set_default_device('cuda')
    do_levels = [8]
    num_samples = 100
    folder = "hyperparameters2"
    
    hp_grid_no_model = {
        "filter_raw_sinogram_with_a": [0.0, 4.5, 5.0, 5.5, 6.0],
        "use_tv_reg": [0.0,0.01, 0.1, 0.5],
        "use_autoencoder_reg": [0.0, 0.001, 0.05, 0.1, 0.2],
        "learning_rate": [0.2, 0.3, 0.4],
        "autoencoder_patch_size": [15, 20,30,40],
        "num_iters": [100, 300, 600],
        "use_no_model": [True],
        "p_loss": [1],
    }
    
    hp_grid_model = {
        "filter_raw_sinogram_with_a": [0.0, 4.5, 5.0, 5.5, 6.0],
        "use_tv_reg": [0.0,0.01, 0.1, 0.5],
        "use_autoencoder_reg": [0.0, 0.001, 0.05, 0.1, 0.2],
        "learning_rate": [0.001, 0.005, 0.01],
        "autoencoder_patch_size": [15, 20,30,40],
        "num_iters": [400],#, 800, 1200],
        "use_no_model": [False],
        "p_loss": [1],
    }
    
    # Create a list of all possible combinations of hyperparameters
    hp_combinations_no_model = list(itertools.product(*hp_grid_no_model.values()))
    hp_combinations_model = list(itertools.product(*hp_grid_model.values()))
    print(f"Number of combinations for model: {len(hp_combinations_model)}")
    print(f"Number of combinations for no model: {len(hp_combinations_no_model)}")
    
    # Create a list of all possible combinations of hyperparameters
    hp_combinations = hp_combinations_model + hp_combinations_no_model
    print(f"Total number of combinations: {len(hp_combinations)}")
    
    # Take a random sample of the combinations
    np.random.shuffle(hp_combinations)
    hp_combinations = hp_combinations[:num_samples]
    for i,combination in tqdm(enumerate(hp_combinations)):
        params = {k:v for k,v in zip(hp_grid_model.keys(), combination)}
        params = set_default_params(params, overwrite=False)
        results = test_parameters(params, levels=do_levels, overwrite=False, folder_name=folder)
        print(f"Finished {i+1}/{num_samples} combinations")
    print("All done!")
    
    
    
    