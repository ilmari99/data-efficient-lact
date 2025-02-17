import torch as pt
import torch.nn as nn
import numpy as np
import itertools
import torchvision
from pytorch_models import HTCModel
from utils import FBPRadon



class ReconstructorBase(nn.Module):
    """
    Takes in a sinogram, and outputs y_hat and s_hat.
    This level should handle all data formatting, setting the image_mask, and edge_masks.
    """
    def __init__(self, proj_dim : int,
                 angles,
                 a = 0.1,
                 image_mask = None,
                 device=None,
                 edge_pad_size=5,
                 ):
        super(ReconstructorBase, self).__init__()
        self.dim = proj_dim
        self.angles = np.array(angles)
        self.output_image_shape = (self.dim, self.dim)
        self.device = device
        if image_mask is None:
            image_mask = pt.ones(self.output_image_shape, device=device, requires_grad=False)
        self.image_mask = image_mask
        self.edge_pad_mask = self.get_edge_padding_mask(image_mask, pad_size=edge_pad_size)
        self.radon_t = FBPRadon(proj_dim, self.angles, a = a, clip_to_circle=False, device=device)
        
    def get_edge_padding_mask(self, image_mask, pad_size=1):
        """ Returns a mask of size dim, dim.
        """
        if image_mask is None or pad_size==0:
            return pt.zeros((self.dim, self.dim), device='cuda', requires_grad=False)
        # Scale the mask down by pad_size
        scaled_down_image_mask = pt.nn.functional.interpolate(image_mask.unsqueeze(0).unsqueeze(0), size=(self.dim-pad_size, self.dim-pad_size), mode='nearest')
        scaled_down_image_mask = scaled_down_image_mask.squeeze()
        print(f"Scaled down image mask shape: {scaled_down_image_mask.shape}")
        # Pad to dim, dim
        pu_pl = pad_size // 2
        pd_pr = pad_size - pu_pl
        scaled_down_image_mask_padded = pt.nn.functional.pad(scaled_down_image_mask, (pu_pl, pd_pr, pu_pl, pd_pr))
        print(f"Scaled down image mask padded shape: {scaled_down_image_mask_padded.shape}")
        # Now, to get the edge_padding_mask, we take a zerto matrix, and set all pixels to 1,
        # where the scaled_down_image_mask_padded is 0 AND where the original image_mask is 1
        edge_pad_mask = pt.zeros((self.dim, self.dim), device='cuda', requires_grad=False)
        edge_pad_mask[(scaled_down_image_mask_padded == 0) & (image_mask == 1)] = 1
        return edge_pad_mask
    
    def forward(self, s):
        raise NotImplementedError("Forward method not implemented")
    
    def parameters(self):
        raise NotImplementedError("Parameters method not implemented")

class NoModel(ReconstructorBase):
    """ No model, where we just optimize a collection of weights in a matrix
    (image of a filled circle) to produce the sinogram.
    """
    def __init__(self, proj_dim, angles, a=0.1, image_mask=None, device="cuda", edge_pad_size=5, scale_sinogram=False, shift_mask=False):
        super(NoModel, self).__init__(proj_dim, angles, a=a, image_mask=image_mask, device=device, edge_pad_size=edge_pad_size)
        if scale_sinogram:
            raise NotImplementedError("Scaling sinogram not implemented for NoModel")
        self.shift_mask = shift_mask
        self.weights = pt.tensor(self.image_mask, device=self.device, requires_grad=True)
        self.mask_offset = pt.tensor([0.0, 0.0], device=self.device, requires_grad=True)

    def parameters(self):
        return [self.weights, self.mask_offset] if self.shift_mask else [self.weights]

    def forward(self, _):
        y_hat = pt.sigmoid(self.weights)

        # Create an identity affine transformation
        theta = pt.eye(2, 3, device=self.device, dtype=pt.float32).unsqueeze(0)
        
        if self.shift_mask:
            # Update the translation part with mask_offset in a differentiable way
            theta[:, 0, 2] = self.mask_offset[0]
            theta[:, 1, 2] = self.mask_offset[1]
            mask_shape = self.image_mask.unsqueeze(0).unsqueeze(0).shape
            grid = pt.nn.functional.affine_grid(theta, mask_shape, align_corners=False)
            shifted_mask = pt.nn.functional.grid_sample(
                self.image_mask.unsqueeze(0).unsqueeze(0).float(),
                grid, align_corners=False
            ).squeeze()
        else:
            shifted_mask = self.image_mask

        # Multiply y_hat with the shifted image mask.
        y_hat = y_hat * shifted_mask

        # Set all pixels that are in the edge_pad to 1.
        y_hat = pt.where(self.edge_pad_mask == 1, pt.tensor(1.0, device=self.device), y_hat)
        s_hat = self.radon_t.forward(y_hat)
        return y_hat, s_hat
        
class HTCModelReconstructor(ReconstructorBase):
    """ This class is for models that predict the image directly from the sinogram.
    """
    def __init__(self, proj_dim: int, angles, a=0.1, image_mask=None, device="cuda", edge_pad_size=5, init_features=32, shift_mask=False):
        self.shift_mask = shift_mask
        self.use_original = True if proj_dim == 560 else False
        self.init_features = init_features
        super().__init__(proj_dim, angles, a=a, image_mask=image_mask, device=device, edge_pad_size=edge_pad_size)
        self.nn = self.create_nn()
        
        # Initialize the offset parameters (translation in x and y) for shifting the image mask.
        self.mask_offset = pt.tensor([0.0, 0.0], device=self.device, requires_grad=True)
    
    def create_nn(self) -> nn.Module:
        if self.use_original:
            print("Using original HTC model")
            htc_net = HTCModel((181, self.dim), init_features=self.init_features, overwrite_cache=False, init_channels=2)
        else:
            print("Using new HTC model")
            htc_net = HTCModel((len(self.angles), self.dim), init_features=self.init_features, overwrite_cache=False, init_channels=1)
            
        htc_net.to(self.device)
        return htc_net
    
    def parameters(self):
        # Include both the network parameters and the mask_offset parameter.
        return itertools.chain(self.nn.parameters(), [self.mask_offset]) if self.shift_mask else self.nn.parameters()
    
    def forward(self, s):
        s = s.float().to(self.device)
        if self.use_original:
            s = s / 255
            # Pad the sinogram to 181 x 512.
            missing_rows = 181 - s.shape[0]
            s = pt.nn.functional.pad(s, (0, 0, 0, missing_rows))
            s = s.reshape((1, 1, 181, self.dim))
            # Create a mask that is 1 where we have data, and 0 where we padded.
            mask = pt.ones((181, 560), device=self.device, requires_grad=False)
            mask[181 - missing_rows:, :] = 0
            mask = mask.reshape((1, 1, 181, self.dim))
            # Concatenate mask on the channel dimension.
            s = pt.cat([s, mask], dim=1)
        else:
            s = s.reshape((1, 1, len(self.angles), self.dim))
        
        y_hat = self.nn(s)
        
        if self.use_original:
            # Rotate to angles[0].
            y_hat = torchvision.transforms.functional.rotate(y_hat, np.rad2deg(self.angles[0]))
            y_hat = y_hat.reshape((512, 512))
        
        y_hat = pt.squeeze(y_hat)
        
        # Pad equally from all sides to reach dim x dim.
        num_missing_cols = self.dim - y_hat.shape[1]
        num_missing_rows = self.dim - y_hat.shape[0]
        row_pad_up = num_missing_rows // 2
        row_pad_down = num_missing_rows - row_pad_up
        col_pad_left = num_missing_cols // 2
        col_pad_right = num_missing_cols - col_pad_left
        y_hat = pt.nn.functional.pad(y_hat, (col_pad_left, col_pad_right, row_pad_up, row_pad_down))
        
        if self.shift_mask:
            # Shift the image mask using the trainable offset parameters.
            theta = pt.eye(2, 3, device=self.device, dtype=pt.float32).unsqueeze(0)
            theta[:, 0, 2] = self.mask_offset[0]
            theta[:, 1, 2] = self.mask_offset[1]
            mask_shape = self.image_mask.unsqueeze(0).unsqueeze(0).shape
            grid = pt.nn.functional.affine_grid(theta, mask_shape, align_corners=False)
            shifted_mask = pt.nn.functional.grid_sample(
                self.image_mask.unsqueeze(0).unsqueeze(0).float(), grid, align_corners=False
            ).squeeze()
        else:
            shifted_mask = self.image_mask
        
        # Multiply the reconstruction by the shifted mask.
        y_hat = y_hat * shifted_mask
        
        # Set all pixels in the edge_pad to 1.
        y_hat = pt.where(self.edge_pad_mask == 1, pt.tensor(1.0, device=self.device), y_hat)
        y_hat = y_hat.squeeze()
        s_hat = self.radon_t.forward(y_hat)
        return y_hat, s_hat