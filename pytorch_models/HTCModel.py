import itertools
import json
import os

import torch
from torch import nn

# CREDIT: https://github.com/99991/HTC2022-TUD-HHU-version-1

class HTCModel(nn.Module):
    def __init__(self, input_shape, init_features=32, overwrite_cache=False, load_weights="", init_channels=2):
        """
        Initializes the HTCModel class. THis class uses a convolutional
        autoencoder to reconstruct the shape of an object from thickness
        measurements.
        Args:
            input_shape (tuple): The shape of the input tensor (B, 1, N, D).
            init_features (int, optional): The number of initial features. Defaults to 32.
            overwrite_cache (bool, optional): Whether to overwrite the cache. Defaults to False.
            load_weights (str, optional): Path to the weights file to load. Defaults to "".
            init_channels (int, optional): The number of initial channels. Defaults to 2.
        """
        
        super().__init__()
        assert init_features >= 4, "init_features must be at least 8"
        assert init_features % 4 == 0, "init_features must be divisible by 4"
        #assert input_shape[1] == 512, "Input shape must have 512 columns"


        def blocks(dim, n):
            return [Block(dim) for _ in range(n)]
        
        # Original parameters
        encoder_kernel1 = (4, 4)
        encoder_stride1 = (4, 4)
        encoder_padding1 = (1, 1)
        encoder_kernel2 = (3, 4)
        encoder_stride2 = (3, 4)
        encoder_padding2 = (1, 1)
        encoder_kernel3 = (3, 4)
        encoder_stride3 = (2, 4)
        encoder_padding3 = (1, 0)
        
        # We might want to use different parameters
        # in which case we have to find suitable parameters
        # Calculate the output shape of the encoder
        # The input shape is (B, 1, N, D)
        # [(W−K+2P)/S]+1
        N = input_shape[0]
        D = input_shape[1]
        
        in_cache = self.check_cache(N, D, 8)
        if in_cache and not overwrite_cache:
            (encoder_kernel1,
             encoder_stride1,
             encoder_padding1,
             encoder_kernel2,
             encoder_stride2,
             encoder_padding2,
             encoder_kernel3,
             encoder_stride3,
             encoder_padding3) = in_cache
        else:
            (encoder_kernel1,
             encoder_stride1,
             encoder_padding1,
             encoder_kernel2,
             encoder_stride2,
             encoder_padding2,
             encoder_kernel3,
             encoder_stride3,
             encoder_padding3) = self.find_kernel_stride_padding(N, D, 8)
        
        
        #print(f"Encoder kernel1: {encoder_kernel1}")
        #print(f"Encoder stride1: {encoder_stride1}")
        #print(f"Encoder padding1: {encoder_padding1}")
        #print(f"Encoder kernel2: {encoder_kernel2}")
        #print(f"Encoder stride2: {encoder_stride2}")
        #print(f"Encoder padding2: {encoder_padding2}")
        #print(f"Encoder kernel3: {encoder_kernel3}")
        #print(f"Encoder stride3: {encoder_stride3}")
        
        # Output from encoder should be (B, 4*init_features, 8, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(init_channels, init_features, encoder_kernel1, encoder_stride1, encoder_padding1),
            #nn.Conv2d(1, init_features, (4, 4), (2, 4), (1, 1)),
            nn.BatchNorm2d(init_features),
            *blocks(init_features, n=1),
            nn.BatchNorm2d(init_features),
            nn.Conv2d(init_features, init_features*2, encoder_kernel2, encoder_stride2, encoder_padding2),
            #nn.Conv2d(init_features, init_features*2, (3, 4), (2, 4), (1, 1)),
            *blocks(init_features*2, n=3),
            nn.BatchNorm2d(init_features*2),
            nn.Conv2d(init_features*2, init_features*4, encoder_kernel3, encoder_stride3, encoder_padding3),
            # If input is 60 x 512 -> (B, 4*init_features, 3, 8)
        )
        
        
        self.decoder = nn.Sequential(
            *blocks(init_features*4, n=6),
            nn.BatchNorm2d(init_features*4),
            nn.ConvTranspose2d(init_features*4, init_features*2, 2, 2, 0),
            
            *blocks(init_features*2, n=3),
            nn.ConvTranspose2d(init_features*2, init_features, 2, 2, 0),
            
            *blocks(init_features, n=3),
            nn.ConvTranspose2d(init_features, init_features//4, 2, 2, 0),
            
            *blocks(init_features//4, n=1),
            
            nn.ConvTranspose2d(init_features//4, init_features//8, 2, 2, 0),
            *blocks(init_features//8, n=1),
            nn.ConvTranspose2d(init_features//8, init_features//8, 2, 2, 0),
            *blocks(init_features//8, n=1),
            
            nn.ConvTranspose2d(init_features//8, init_features//8, 2, 2, 0),
            *blocks(init_features//8, n=1),
            nn.Conv2d(init_features//8, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        
        if load_weights:
            try:
                self.load_state_dict(torch.load(load_weights))
            except Exception as e:
                print("Could not load weights: ", e)
        
    def check_cache(self, N, D, output_shape):
        if not os.path.exists("htc_model_params_cache.json"):
            return None
        with open("htc_model_params_cache.json", "r") as f:
            content = f.read()
            if content:
                content = json.loads(content)
                #print(f"Content: {content}")
                # Convert keys to tuples
                content = {eval(k):v for k,v in content.items()}
                #print(f"Content: {content}")
                if (N, D, output_shape) in content:
                    data = content[(N, D, output_shape)]
                    return tuple(data.values())
        return None
        
    def find_kernel_stride_padding(self, N, D, output_shape):
        grid = {
            "nk1":list(range(1, 6)),
            "ns1":list(range(1, 6)),
            "np1":list(range(0, 2)),
            "dk1":list(range(1,6)),
            "ds1":list(range(1, 6)),
            "dp1":list(range(0,2)),
            "nk2":list(range(1,6)),
            "ns2":list(range(1, 6)),
            "np2":list(range(0,2)),
            "dk2":list(range(1,6)),
            "ds2":list(range(1, 6)),
            "dp2":list(range(0,2)),
            "nk3":list(range(1,6)),
            "ns3":list(range(1, 6)),
            "np3":list(range(0,2)),
            "dk3":list(range(1,6)),
            "ds3":list(range(1, 6)),
            "dp3":list(range(0,2)),
        }
        
        # Loop through all possible combinations of kernel sizes, strides, and paddings
        for params in itertools.product(*grid.values()):
            nk1, ns1, np1, dk1, ds1, dp1, nk2, ns2, np2, dk2, ds2, dp2, nk3, ns3, np3, dk3, ds3, dp3 = params
            n_output_shape = self.calc_encoder_output_shape(N, nk1, ns1, np1, nk2, ns2, np2, nk3, ns3, np3)
            d_output_shape = self.calc_encoder_output_shape(D, dk1, ds1, dp1, dk2, ds2, dp2, dk3, ds3, dp3)
            if n_output_shape == output_shape and d_output_shape == output_shape:
                encoder_kernel1 = (nk1, dk1)
                encoder_stride1 = (ns1, ds1)
                encoder_padding1 = (np1, dp1)
                encoder_kernel2 = (nk2, dk2)
                encoder_stride2 = (ns2, ds2)
                encoder_padding2 = (np2, dp2)
                encoder_kernel3 = (nk3, dk3)
                encoder_stride3 = (ns3, ds3)
                encoder_padding3 = (np3, dp3)
                break
        
        # The content is dictionary, where keys are (N, D, output_shape) tuples,
        # and values are dictionaries with the kernel, stride, and padding values
        with open("htc_model_params_cache.json", "r") as f:
            content = f.read()
            if not content:
                content = "{}"
            content = json.loads(content)
            #print(f"Content: {content}")
        # cache
        with open("htc_model_params_cache.json", "w") as f:
        
            entry = {f"({N}, {D}, {output_shape})": {
                "encoder_kernel1":encoder_kernel1,
                "encoder_stride1":encoder_stride1,
                "encoder_padding1":encoder_padding1,
                "encoder_kernel2":encoder_kernel2,
                "encoder_stride2":encoder_stride2,
                "encoder_padding2":encoder_padding2,
                "encoder_kernel3":encoder_kernel3,
                "encoder_stride3":encoder_stride3,
                "encoder_padding3":encoder_padding3
            }}
            
            # Update content and write it back to the file
            content.update(entry)
            print(f"Writing content: {content}")
            f.write(json.dumps(content))
        return encoder_kernel1, encoder_stride1, encoder_padding1, encoder_kernel2, encoder_stride2, encoder_padding2, encoder_kernel3, encoder_stride3, encoder_padding3
        
    def calc_encoder_output_shape(self, d0, k1, s1, p1, k2, s2, p2, k3, s3, p3):
        #(((((d_0 - k_1 + 2p_1)/s_1 + 1) - k_2 + 2*p_2)/s_2 + 1) - k_3 +2*p_3) / s_3 + 1
        if (d0 - k1 + 2*p1) % s1 != 0:
            return 0
        if ((d0 - k1 + 2*p1) / s1 + 1 - k2 + 2*p2) % s2 != 0:
            return 0
        if ((((d0 - k1 + 2*p1)//s1 + 1 - k2 + 2*p2)//s2 + 1 - k3 + 2*p3) % s3) != 0:
            return 0
        return ((((d0 - k1 + 2*p1)//s1 + 1 - k2 + 2*p2)//s2 + 1 - k3 + 2*p3)//s3 + 1)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    
class Block(nn.Module):
    def __init__(self, dim, kernel_size=5, expansion=2):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = kernel_size // 2
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, padding)
        self.norm = nn.BatchNorm2d(dim)
        hidden_dim = int(expansion * dim)
        self.ln1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = nn.GELU()
        self.ln2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ln1(out)
        out = self.act(out)
        out = self.ln2(out)
        out += x
        return out