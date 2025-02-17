import numpy as np
import torch
from kornia.filters import spatial_gradient
from utils import PatchAutoencoder

def create_autoencoder_regularization(autoencoder_path, patch_size, num_latent, stride, batch_size):
    """ Return a callable that can be used to regularize the latent representation of an image.
    """
    autoencoder = PatchAutoencoder(patch_size, num_latent, autoencoder_path)
    # Set the dtype to float16 and mode to eval
    autoencoder.to('cuda', dtype=torch.float32)
    autoencoder = autoencoder.eval()

    def regularization(y_hat):
        # Remove noise from the image
        reconstruction = autoencoder.remove_noise_from_img(y_hat,
                                                 patch_size=patch_size,
                                                 stride=stride,
                                                 batch_size=batch_size,
                                                 #patches_to_device='cpu',
                                                 #patches_to_dtype=torch.float16
        )
        reconstruction = reconstruction.to(y_hat.device)
        # Check that there are no NaN values
        assert not torch.isnan(reconstruction).any(), "There are NaN values in the reconstruction"
        assert not torch.isnan(y_hat).any(), "There are NaN values in y_hat"
        diff = torch.mean(torch.abs(y_hat - reconstruction))
        return diff
    return regularization

def create_autoencoder_regularization_diff(autoencoder_path, patch_size, num_latent, stride, batch_size):
    """ Return a callable that can be used to regularize the latent representation of an image.
    """
    autoencoder = PatchAutoencoder(patch_size, num_latent, autoencoder_path)
    autoencoder.to('cuda', dtype=torch.float32)
    autoencoder = autoencoder.eval()

    def regularization(y_hat):
        # Remove noise from the image
        reconstruction = autoencoder.remove_noise_from_img_diff(y_hat,
                                                 patch_size=patch_size,
                                                 stride=stride,
                                                 batch_size=batch_size,
        )
        reconstruction = reconstruction.to(y_hat.device)
        # Check that there are no NaN values
        
        assert not torch.isnan(reconstruction).any(), "There are NaN values in the reconstruction"
        assert not torch.isnan(y_hat).any(), "There are NaN values in y_hat"
        diff = torch.mean(torch.abs(y_hat - reconstruction))
        return diff
    return regularization

def total_variation_regularization(mat, normalize=True, order=1):
    """ Calculate the total variation of an image
    """
    # Calculate the gradient
    grad = spatial_gradient(mat.reshape(1,1,mat.shape[0],mat.shape[1]), order=order)
    grad = grad.squeeze()
    # Check that there are no NaN values
    assert not torch.isnan(grad).any(), "There are NaN values in the gradient"
    grad_x = grad[0,:,:]
    grad_y = grad[1,:,:]
    # Anisotropic TV, since we need differentiability
    grad_x_abs = torch.abs(grad_x)
    grad_y_abs = torch.abs(grad_y)
    tv = torch.sum(grad_x_abs + grad_y_abs)
    if normalize:
        tv = tv / (mat.shape[0] * mat.shape[1])
    return tv

def tikhonov_regularization(mat, normalize=True):
    """ Penalize the norm of the gradient of the image
    """
    #print(mat)
    # Calculate the gradient
    grad = spatial_gradient(mat.reshape(1,1,mat.shape[0],mat.shape[1]))
    grad = grad.squeeze()
    #assert not torch.isnan(grad).any(), "There are NaN values in the gradient"
    grad_x = grad[0,:,:]
    grad_y = grad[1,:,:]
    # Calculate the norm of the gradient
    norm = torch.sum(grad_x**2 + grad_y**2)
    if normalize:
        norm = norm / (mat.shape[0] * mat.shape[1])
    return norm


def vector_similarity_regularization(y_hat, base_images, coeff=0.01):
    """
    Calculate the average cosine similarity between y_hat and base_images,
    after converting them to vectors.
    """
    model = mobilenet_v3_small(pretrained=True)
    model.eval()
    model.to('cuda')
    # Get the output of the first layer of the VGG19 model
    vgg_output = model.features[:-1](y_hat)
    vgg_output_base = model.features[:-1](base_images)
    #print(f"VGG output shape: {vgg_output.shape}")
    # Flatten the output
    vgg_output = vgg_output.view(vgg_output.size(0), -1)
    vgg_output_base = vgg_output_base.view(vgg_output_base.size(0), -1)
    # Calculate average cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(vgg_output, vgg_output_base)
    #print(cosine_similarity)
    return cosine_similarity

def make_edge_filter(size=3):
    """ Make a filter that can be used to calculate the number of edges in an image.
    """
    # Only square filters are supported
    assert size % 2 == 1, "Size must be odd"
    filter_ = np.ones((size, size))
    # The middle element is size^2 - 1
    filter_[size//2, size//2] = size**2 - 1
    #print(f"Filter: {filter_}")
    filter_ = torch.tensor(filter_, device='cuda', dtype=torch.float32, requires_grad=True)
    return filter_

def number_of_edges_regularization(y_hat, filter_sz=3, coeff=0.1):
    """ If y_hat has many pixels that are not surrounded by similar pixels, penalize it.
    """
    # First, convert y_hat to a binary image
    #y_hat = y_hat > 0.5
    # Set all 0s to -1
    y_hat = y_hat * 2 - 1
    # Count the number of edges
    edge_filter = make_edge_filter(size=filter_sz)
    filter_sz = torch.tensor(filter_sz, device='cuda', dtype=torch.float32)
    # Convolve y_hat with the edge filter
    y_hat = y_hat.unsqueeze(0).unsqueeze(0)
    similarity_matrix = torch.nn.functional.conv2d(y_hat, edge_filter.unsqueeze(0).unsqueeze(0))
    similarity_matrix = similarity_matrix.squeeze()
    # Absolute value
    similarity_matrix = torch.abs(similarity_matrix)
    # Divide by (size^2 - 1)*2, to scale it to be between 0 and 1
    similarity_matrix = similarity_matrix / ((filter_sz**2 - 1)*2)
    similarity_matrix = 1-similarity_matrix
    # clip to 0 and 1
    similarity_matrix = torch.clip(similarity_matrix, 0, 1)
    #print(f"Max similarity: {torch.max(similarity_matrix)}")
    #print(f"Min similarity: {torch.min(similarity_matrix)}")
    #similarity_matrix = torch.tensor(similarity_matrix, device='cuda', dtype=torch.float32, requires_grad=True)
    return similarity_matrix

def ssim_regularization(y_hat, base_images, coeff=0.01):
    """
    Calculate the SSIM regularization loss.
    """
    y_hat_replicated = torch.broadcast_to(y_hat, base_images.shape)
    ssim_loss = 1 - ssim(y_hat_replicated, base_images, data_range=1, size_average=False)#, nonnegative_ssim=True, win_size=15, win_sigma=7.5)
    return torch.mean(ssim_loss) * coeff

def binary_regularization(y_hat, norm=1):
    """ Penalize y_hat based on how different it is from it's binary version
    """
    assert norm in [1, 2], "Only L1 and L2 norms are supported"
    y_hat_bin = y_hat > 0.5
    # Cast to float32
    y_hat_bin = y_hat_bin.float()
    diff = y_hat - y_hat_bin
    if norm == 1:
        return torch.mean(torch.abs(diff))
    return torch.mean(diff**2)
    
    

def histogram_regularization(y_hat, true_y):
    """ Penalize based on how different the histogram of y_hat is from the histogram of true_y
    """
    y_hat = y_hat.reshape((128,128))
    true_y = true_y.reshape((128,128))
    y_hat_hist = torch.histc(y_hat, bins=10, min=0, max=1)
    true_y_hist = torch.histc(true_y, bins=10, min=0, max=1)
    # Compare the distributions using chi-squared distance
    chi_squared = torch.sum((y_hat_hist - true_y_hist)**2 / (y_hat_hist + true_y_hist))
    # Scale to be between 0 and 1 (so 1 == 1000)
    return torch.clip(chi_squared / 1000, 0, 1)