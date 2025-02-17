import os
import random
import sys
import torch
import numpy as np
from torchvision.io import read_image, ImageReadMode
from PIL import Image
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt
from utils import PatchAutoencoder

def extract_patches_2d_pt(image, patch_size=8,device=None, dtype=None, stride=1):
    """ A Pytorch implementation of extract_patches_2d.
    It takes in an image (WxH) and returns all the (patch_sz x patch_sz) patches with stride=1.
    """
    # Add batch and channel dimensions
    image = image.unsqueeze(0).unsqueeze(0)
    patches = torch.nn.functional.unfold(image, patch_size, stride=stride)
    patches = patches.permute(0, 2, 1).reshape(-1, patch_size, patch_size)
    patches = patches.to(device, dtype=dtype)
    return patches

def reconstruct_from_patches_2d_pt(patches, image_size, device=None, stride=1):
    """ A Pytorch implementation of reconstruct_from_patches_2d.
    It takes in patches, and reconstructs the image, averaging overlapping patches
    """
    patch_size = patches.shape[-1]
    
    #print(f"Patches shape: {patches.shape}")
    patches = patches.reshape(-1, patch_size*patch_size)
    patches = patches.unsqueeze(0).permute(0, 2, 1)
    #print(f"Patches shape: {patches.shape}")
    counter_image = torch.ones_like(patches, device=device)
    #counter_image = torch.nn.functional.unfold(counter_image.unsqueeze(0).unsqueeze(0), patch_size, stride=1)
    #print(f"Counter image shape: {counter_image.shape}")
    # Fold patches of ones to know how many patches overlap each pixel
    counter_image = torch.nn.functional.fold(counter_image, image_size, patch_size, stride=stride)
    counter_image = counter_image.squeeze()
    #print(f"Counter image shape: {counter_image.shape}")
    # fold the patches
    image = torch.nn.functional.fold(patches, image_size, patch_size, stride=stride)
    image = image.squeeze()
    # Divide the image by the counter image
    image = image / counter_image
    # nans to zeros
    image = torch.nan_to_num(image)
    image = image.to(device)
    return image

def shuffle_local_pixels(image, area=16, shuffle_chance=0.5):
    """ Shuffle the pixels in a local area of the image
    """
    shuffled_image = image.copy()
    for i in range(0, image.shape[0], area):
        for j in range(0, image.shape[1], area):
            if np.random.rand() < shuffle_chance:
                # Shuffle the pixels in the torch image
                min_i_idx = i
                max_i_idx = min(i + area, image.shape[0])
                min_j_idx = j
                max_j_idx = min(j + area, image.shape[1])
                # Get the pixels
                pixels = shuffled_image[min_i_idx:max_i_idx, min_j_idx:max_j_idx]
                # Shuffle the pixels
                pixels = pixels.flatten()
                np.random.shuffle(pixels)
                pixels = pixels.reshape(max_i_idx-min_i_idx, max_j_idx-min_j_idx)
                shuffled_image[min_i_idx:max_i_idx, min_j_idx:max_j_idx] = pixels
                
    return shuffled_image

def resize_distort(image, factor=0.5):
    """ Smooth the pixels of the image by resizing it and then resizing it back.
    """
    og_shape = image.shape
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0).unsqueeze(0)
    image = torch.nn.functional.interpolate(image, scale_factor=factor, mode='bilinear')
    image = torch.nn.functional.interpolate(image, mode='bilinear',size=og_shape)
    if image.device.type == 'cuda':
        image = image.cpu()
    image = image.squeeze().numpy()
    return image

def load_generated_images(path,device=None):
    images = []
    for image_path in os.listdir(path):
        if image_path.endswith(".png"):
            img = read_image(os.path.join(path, image_path), mode=ImageReadMode.GRAY)
            img = torch.tensor(img, dtype=torch.float32, device=device)
            img = img.squeeze() / 255
            # Round to 0 or 1
            img = torch.round(img)
            images.append(img)
    return images

def get_htc_scan(level = 1, sample = "a"):
    base_path = "./htc2022_data/"
    htc_file = f"htc2022_0{level}{sample}_recon_fbp_seg.png"
    print(f"Loading sample {level}{sample}")
    # Read image
    img = Image.open(os.path.join(base_path, htc_file))
    img = np.array(img, dtype=np.float32)
    max_val = np.max(img)
    img = img / max_val
    return img


def training_loop(autoenc,
                  train_filtered_patches,
                  test_filtered_patches,
                  num_epochs=5,
                  patience=10,
                  learning_rate=0.001,
                  batch_size=64,
                  restore_best=True,
                  save_to="patch_autoencoder.pth",
                  overwrite=False):
    # Train the autoencoder
    optimizer = torch.optim.Adam(autoenc.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()
    
    # Naming
    writer = SummaryWriter(f"ae_training/P{patch_size}_D{num_latent}")
    
    true_image = get_htc_scan(8, "a")

    # Move to device
    autoenc = autoenc.to(torch.get_default_device())
    #train_filtered_patches = train_filtered_patches.to(torch.get_default_device())
    #test_filtered_patches = test_filtered_patches.to(torch.get_default_device())
    
    
    # Train the autoencoder
    best_loss = np.inf
    best_epoch = 0
    best_weights = None
    for epoch in range(num_epochs):
        train_filtered_patches = train_filtered_patches[torch.randperm(len(train_filtered_patches),device="cpu")]
        for i in range(0, len(train_filtered_patches), batch_size):
            batch = train_filtered_patches[i:i+batch_size]
            batch = torch.tensor(batch, dtype=torch.float32).to(torch.get_default_device())
            optimizer.zero_grad()
            decoded = autoenc(batch)
            decoded = decoded.squeeze(1)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss}", end="\r")
        writer.add_scalar("Loss/train", loss, epoch)
        # Test the model
        with torch.no_grad():
            for i in range(0, len(test_filtered_patches), batch_size):
                batch = test_filtered_patches[i:i+batch_size]
                batch = torch.tensor(batch, dtype=torch.float32).to(torch.get_default_device())
                decoded = autoenc(batch)
                decoded = decoded.squeeze(1)
                test_loss = criterion(decoded, batch)
            # Test noise removal
            distorted_true_image = true_image.copy()
            distorted_true_image = shuffle_local_pixels(distorted_true_image, area=patch_size//2, shuffle_chance=0.4)
            distorted_true_image = resize_distort(distorted_true_image, factor=0.2)
            reconstructed = autoenc.remove_noise_from_img(torch.tensor(distorted_true_image, dtype=torch.float32),
                                                            patch_size,
                                                            stride = patch_size//8,
                                                            batch_size=batch_size,
                                                            )
            mae = np.mean(np.abs(true_image - reconstructed.cpu().numpy()))
            print(f"Epoch {epoch}, Test loss: {test_loss}, MAE: {mae}")
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("MAE", mae, epoch)
            # Save image and reconstruction
            if epoch == 0:
                writer.add_image("Original", torch.tensor(true_image).unsqueeze(0), epoch)
            writer.add_image("Distorted", torch.tensor(distorted_true_image).unsqueeze(0), epoch)
            writer.add_image("Reconstructed", reconstructed.unsqueeze(0), epoch)
            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch
                best_weights = autoenc.state_dict()
            if epoch - best_epoch > patience:
                break
    # Restore the best weights
    if restore_best:
        autoenc.load_state_dict(best_weights)
    # Save the model
    if overwrite or not os.path.exists(save_to):
        torch.save(autoenc.state_dict(), save_to)
    elif not overwrite:
        print(f"Model file {save_to} already exists, not overwriting")
    return autoenc


if __name__ == "__main__":
    patch_size = 32
    num_epochs = 50
    patience = 5
    restore_best = True
    learning_rate = 0.0001
    batch_size = 64
    train_test_split = 0.85
    load_pre_trained = ""
    do_training = True
    overwrite = False
    test_stride = 1
    from_command_line = True if len(sys.argv) > 1 else False
    do_demo = True if not from_command_line else False
    
    parser = argparse.ArgumentParser(description='Patch Autoencoder')
    parser.add_argument('--patch_size', type=int, default=30, help='Size of the patches')
    args = parser.parse_args()
    if from_command_line:
        patch_size = args.patch_size
    num_latent = patch_size // 4
    train_stride = patch_size // 5
    
    save_to = f"patch_autoencoder_P{patch_size}_D{num_latent}.pth"
    
    #torch.manual_seed(0)
    
    
    # Check that GPU is available
    if torch.cuda.is_available():
        print("Using GPU")
        torch.set_default_device('cuda')
    else:
        torch.set_default_device('cpu')
    
    # Load images as numpy arrays, and use sklearn to learn the dictionary.
    #images = [get_basic_circle_image() for _ in range(10)]
    images = load_generated_images("generated_data")    
    if images[0].device.type == 'cuda':
        images = [img.cpu().numpy() for img in images]
    if not do_training:
        images = random.choices(images, k = 3)
    print(f"Loading {len(images)} images")
    images = np.array(images)
    
    images = [torch.tensor(img, dtype=torch.float32) for img in images]
    image_patches = [extract_patches_2d_pt(img, patch_size,"cuda",dtype=torch.bool,stride=train_stride) for img in images]
    
    # Concatenate the patches
    all_patches = torch.cat(image_patches, dim=0)
    all_patches = all_patches.to("cpu")
    print(f"All patches shape: {all_patches.shape}")
    
    #exit()
    # Filter out the patches where > 80% of the pixels are zero
    patch_sum = all_patches.sum(dim=(1,2), dtype=torch.float16)
    patch_sum = patch_sum / (patch_size * patch_size)
    filtered_patches = all_patches#[patch_sum > 0.2]
    print(f"Filtered patches shape: {filtered_patches.shape}")
    
    #filtered_patches = filtered_patches.to(torch.get_default_device())
    
    # Split the patches to train and test
    random_indices = torch.randperm(len(filtered_patches), device="cpu")
    train_indices = random_indices[0:int(train_test_split*len(filtered_patches))]
    train_filtered_patches = filtered_patches[train_indices]
    test_filtered_patches = filtered_patches[random_indices[int(train_test_split*len(filtered_patches)):]]
    
    autoenc = PatchAutoencoder(patch_size, num_latent)
    
    autoenc = autoenc.to(torch.get_default_device())
    #train_filtered_patches = train_filtered_patches.to(torch.get_default_device())
    #test_filtered_patches = test_filtered_patches.to(torch.get_default_device())
    
    if load_pre_trained:
        autoenc.load_state_dict(torch.load(load_pre_trained))
        
    if do_training:
        autoenc = training_loop(autoenc,
                                train_filtered_patches,
                                test_filtered_patches,
                                num_epochs=num_epochs,
                                patience=patience,
                                learning_rate=learning_rate,
                                batch_size=batch_size,
                                restore_best=restore_best,
                                save_to=save_to,
                                overwrite=overwrite)
    
    if not do_demo:
        print(f"Training done, saved model to {save_to}")
        sys.exit(0)
    
    # Show 4 enc-dec examples
    fig, ax = plt.subplots(2, 4)
    # Make a big suptitle
    fig.suptitle(f"Patch autoencoder with patch size {patch_size} and {num_latent} latent variables",
                    fontsize=16)
    with torch.no_grad():
        for i in range(4):
            idx = np.random.randint(0, len(test_filtered_patches))
            original = test_filtered_patches[idx].clone().squeeze().cpu().numpy()
            original = original.astype(np.float32)
            while np.sum(original) < 0.1 * patch_size * patch_size or np.sum(original) > 0.8 * patch_size * patch_size:
                idx = np.random.randint(0, len(test_filtered_patches))
                original = test_filtered_patches[idx].clone().squeeze().cpu().numpy()
                original = original.astype(np.float32)
            original_distorted = original.copy()
            if i >= 2:
                # Distort the patch
                original_distorted = shuffle_local_pixels(original_distorted, area=patch_size//8, shuffle_chance=0.4)
                original_distorted = resize_distort(original_distorted, factor=0.4)
            ax[0,i].matshow(original_distorted)
            ax[0,i].set_title("Input patch" if i < 2 else "Input patch\nwith artifacts")
            
            
            
            enc = autoenc.encode(torch.tensor(original_distorted).unsqueeze(0).unsqueeze(0))
            dec = autoenc.decoder(enc)
            
            ax[1,i].matshow(dec.squeeze().cpu().numpy())
            latent = enc.squeeze().cpu().numpy()
            mae = np.mean(np.abs(original - dec.squeeze().cpu().numpy()))
            ax[1,i].set_title(f"Encoded-Decoded\nMAE: {mae:.2f}")
            
            for a in ax[:,i]:
                a.set_xticks([])
                a.set_yticks([])
            
        # Load an image
        img = get_htc_scan(7, "a")
        img_distorted = img
        img_distorted = shuffle_local_pixels(img, area=patch_size // 2, shuffle_chance=0.4)
        img_distorted = resize_distort(img_distorted, factor=0.2)
        
        img = torch.tensor(img, dtype=torch.float32)
        img_distorted = torch.tensor(img_distorted, dtype=torch.float32)
        print(f"Distorted image shape: {img_distorted.shape}")
        
        reconstructed = autoenc.remove_noise_from_img(img_distorted,
                                                      patch_size,
                                                      stride = test_stride,
                                                      batch_size=batch_size,
                                                      )
        
        img_cpu_np = img.cpu().numpy()
        img_distorted_cpu_np = img_distorted.cpu().numpy()
        reconstructed_cpu_np = reconstructed.cpu().numpy()
        # The reconstructed image might have nans, replace them with zeros
        reconstructed_cpu_np = np.nan_to_num(reconstructed_cpu_np)
        
        mae = np.mean(np.abs(reconstructed_cpu_np - img_cpu_np))
        print(f"MAE: {mae}")

        fig, ax = plt.subplots(1, 3)
        fig.suptitle(f"Removing artifacts using a {patch_size}x{patch_size}\npatch autoencoder with {num_latent} latent variables",
                    fontsize=16)
        ax[0].matshow(img_cpu_np)
        ax[0].set_title("Original image")
        ax[1].matshow(img_distorted_cpu_np)
        ax[1].set_title("Distorted image")
        ax[2].matshow(reconstructed_cpu_np)
        ax[2].set_title("Reconstructed image")
        # Remove ticks
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        # Show mae of reconstruction
        ax[2].set_title(f"Reconstructed image\nMAE: {mae:.2f}")
    
    
    plt.show()
    
    


    
    
    
    