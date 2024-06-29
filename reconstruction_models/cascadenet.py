import torch
from torch import nn
import fastmri
from fastmri.data import transforms as T

import numpy as np
import torch
import torch.nn as nn

import numpy as np


def gen_mask(kspace, accel_factor=8, seed=None):
    # inspired by https://github.com/facebookresearch/fastMRI/blob/master/common/subsample.py
    shape = kspace.shape
    num_cols = shape[-1]
    print()
    print("num_cols", num_cols)

    center_fraction = (32 // accel_factor) / 100
    print("center_fraction", center_fraction)

    # Create the mask
    num_low_freqs = int(round(num_cols * center_fraction))
    print("num_low_freqs", num_low_freqs)
    prob = (num_cols / accel_factor - num_low_freqs) / (num_cols - num_low_freqs)
    print("prob", prob)
    mask = np.random.default_rng(seed).uniform(size=num_cols) < prob
    print("mask", mask.shape, mask.dtype)
    pad = (num_cols - num_low_freqs + 1) // 2
    print("pad", pad)
    mask[pad:pad + num_low_freqs] = True
    print("mask", mask.shape, mask.dtype)
    # Reshape the mask
    mask_shape = [1 for _ in shape]
    mask_shape[-1] = num_cols
    mask = mask.reshape(*mask_shape)
    print()
    return mask

    
def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out

def enforce_kspace_data_consistency_noiseless(k, k0, mask):

    return (1 - mask) * k + k0

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=5, activation='leaky_relu', n_filters=16):
        super().__init__()
        assert n_convs >= 2, "n_convs must be at least 2"
        
        
        if activation == 'relu':
            self.relu = nn.ReLU
        elif activation == 'leaky_relu':
            self.relu = nn.LeakyReLU
        else:
            raise ValueError(f"Invalid activation: {activation}")

        # layer_list = [
        #     nn.Conv2d(in_channels, n_filters, 3, padding="same", bias=True),
        #     # nn.BatchNorm2d(n_filters),
        #     relu()
        # ]

        # for i in range(n_convs - 2):
        #     layer_list.append(nn.Conv2d(n_filters, n_filters, 3, padding="same", bias=True))
        #     # layer_list.append(nn.BatchNorm2d(n_filters))
        #     layer_list.append(relu())

        
        # layer_list.append(nn.Conv2d(n_filters, out_channels, 3, padding="same", bias=True))
        # # layer_list.append(nn.BatchNorm2d(out_channels))
        # # layer_list.append(relu())

        # self.convs = nn.Sequential(*layer_list)

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(in_channels, n_filters, 3, padding=1, bias=True))

        for i in range(n_convs - 1):
            self.convs.append(nn.Conv2d(n_filters, n_filters, 3, padding=1, bias=True))
        
        self.convs.append(nn.Conv2d(n_filters, out_channels, 3, padding=1, bias=True))

    def forward(self, x):        
        for conv in self.convs:
            x = conv(x)
            x = self.relu()(x)

        return x

class CascadeNet(nn.Module):
    def __init__(self, n_cascade=5, n_convs=5, n_filters=16, noise_lvl=True, activation='relu'):
        super().__init__()
        self.n_cascade = n_cascade
        self.noise_lvl = noise_lvl
        self.cascade = nn.ModuleList()
        for i in range(n_cascade):
            self.cascade.append(ConvBlock(2, 2, n_convs=n_convs, activation=activation, n_filters=n_filters))

    def forward(self, k, m):
        x = fastmri.ifft2c(k)
        for i in range(self.n_cascade):
            x = x.permute(0, 3, 1, 2)
            x_cnn = self.cascade[i](x)
            
            x = x + x_cnn
            x = x.permute(0, 2, 3, 1)
            k_pred = fastmri.fft2c(x)
            k_pred = data_consistency(k_pred, k, m)
            x = fastmri.ifft2c(k_pred)
        return x
            

if __name__ == '__main__':
    net = CascadeNet(n_cascade=5, n_convs=5, n_filters=48, noise_lvl=True, activation='relu')

    # print number of parameters in the model for each layer
    # i have npy files for every layer's weights and biases
    # conv2d_30_weights.npy # cascade.0.convs.0.weight
    # conv2d_30_biases.npy # cascade.0.convs.0.bias
    # conv2d_31_weights.npy # cascade.0.convs.2.weight
    # conv2d_31_biases.npy # cascade.0.convs.2.bias
    # ...
    # conv2d_59_weights.npy # cascade.4.convs.8.weight
    # conv2d_59_biases.npy # cascade.4.convs.8.bias

    # now i need to load these weights and biases into the model
    for cascade in range(5):
        for conv in range(6):
            print("cascade", cascade, "conv", conv, "weights", f"model/conv2d_{30 + cascade * 6 + conv}_weights.npy")
            weight = np.load(f"model/conv2d_{30 + cascade * 6 + conv}_weights.npy")
            bias = np.load(f"model/conv2d_{30 + cascade * 6 + conv}_biases.npy")

            net.cascade[cascade].convs[conv].weight.data = torch.tensor(weight).permute(3, 2, 0, 1)
            net.cascade[cascade].convs[conv].bias.data = torch.tensor(bias)
            # print(f"cascade {cascade} conv {conv} weight", weight.shape, weight.dtype)
            # print(f"cascade {cascade} conv {conv} bias", bias.shape, bias.dtype)
    

    image = np.load("gt_image.npy")
    kspace = np.load("gt_kspace.npy")

    image = torch.tensor(image, dtype=torch.float32)
    kspace = torch.tensor(kspace, dtype=torch.complex64)

    # add batch dimension
    image = image.unsqueeze(0)
    kspace = kspace.unsqueeze(0)

    # print("image", image.shape, image.dtype)
    
    mask = gen_mask(kspace[0, ..., 0], accel_factor=4, seed=0)
    
    fourier_mask = np.repeat(mask.astype(np.float32), kspace.shape[1], axis=0)
    
    

    fourier_mask = torch.from_numpy(fourier_mask[None, ..., None])
    
    masked_kspace = fourier_mask * kspace

    masked_kspace = torch.cat((masked_kspace.real, masked_kspace.imag), dim=-1)
    masked_kspace *= 1e6
    print("masked_kspace", masked_kspace.shape, masked_kspace.dtype)
    print("fourier_mask", fourier_mask.shape, fourier_mask.dtype)
    out = net(masked_kspace, fourier_mask)
    print(out.shape)

    out = fastmri.complex_abs(out)
    print(out.shape)

    out = out.squeeze(0)
    out = out.detach().numpy()

    import matplotlib.pyplot as plt
    plt.imshow(out, cmap='gray')
    plt.savefig("out.png")
    plt.show()


