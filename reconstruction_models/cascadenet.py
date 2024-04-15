import torch
from torch import nn
import fastmri
from fastmri.data import transforms as T

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib

import numpy as np

plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['image.cmap'] = 'gray'
# plt.style.use('ieee')
plt.rcParams['font.size'] = 6


def gen_mask(kspace, accel_factor=8, seed=None):
    # inspired by https://github.com/facebookresearch/fastMRI/blob/master/common/subsample.py
    shape = kspace.shape
    num_cols = shape[-1]

    center_fraction = (32 // accel_factor) / 100
    acceleration = accel_factor

    # Create the mask
    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    mask = np.random.default_rng(seed).uniform(size=num_cols) < prob
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad:pad + num_low_freqs] = True

    # Reshape the mask
    mask_shape = [1 for _ in shape]
    mask_shape[-1] = num_cols
    mask = mask.reshape(*mask_shape)
    return mask

    
def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    print ("########################")
    print("k shape and dtype", k.shape, k.dtype)
    print("k0 shape and dtype", k0.shape, k0.dtype)
    print("mask shape and dtype", mask.shape, mask.dtype)
    print("noise_lvl", noise_lvl)
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out

    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs, activation='leaky_relu'):
        super().__init__()
        layer_list = []
        for i in range(n_convs):
            layer_list.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))
            layer_list.append(nn.BatchNorm2d(out_channels))

            if activation == 'relu':
                layer_list.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layer_list.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Invalid activation: {activation}")

        self.convs = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.convs(x)

class CascadeNet(nn.Module):
    def __init__(self, n_cascade=5, n_convs=5, n_filters=16, noiseless=True, activation='relu'):
        super().__init__()

        self.n_cascade = n_cascade
        self.n_convs = n_convs
        self.n_filters = n_filters
        self.noiseless = noiseless
        self.activation = activation

        self.cascade = nn.ModuleList()

        for i in range(n_cascade):
            self.cascade.append(ConvBlock(2, 2, n_convs, activation))


    def forward(self, k, m):
        print("Initial k shape and dtype", k.shape, k.dtype)
        print("Initial m shape and dtype", m.shape, m.dtype)

        # x = torch.fft.fftshift(torch.fft.ifft2(k))
        x = fastmri.ifft2c(k)
        print("x shape and dtype after ifft2", x.shape, x.dtype)

        # make it channel first
        

        for i in range(self.n_cascade):
            x = x.permute(0, 3, 1, 2)
            print("x shape and dtype after permute", x.shape, x.dtype)

            x_cnn = self.cascade[i](x)
            print("x_cnn shape and dtype after cascade", x_cnn.shape, x_cnn.dtype)

            x = x + x_cnn
            print("x shape and dtype after cascade", x.shape, x.dtype)

            # make it channel last
            x = x.permute(0, 2, 3, 1)
            print("x shape and dtype after permute", x.shape, x.dtype)

            k_pred = fastmri.fft2c(x)
            print("k_pred shape and dtype after fft2", k_pred.shape, k_pred.dtype)

            k_pred = data_consistency(k_pred, k, m)
            print("k_pred shape and dtype after data_consistency", k.shape, k.dtype)

            x = fastmri.ifft2c(k_pred)
            print("x shape and dtype after ifft2", x.shape, x.dtype)


        print("Final x shape and dtype", x.shape, x.dtype)
        return x
            

if __name__ == '__main__':
    net = CascadeNet()
    # print("Number of parameters:", sum(p.numel() for p in net.parameters()))

    image = np.load("gt_image.npy")
    kspace = np.load("gt_kspace.npy")

    # print("image", image.shape, image.dtype)
    # print("kspace", kspace.shape, kspace.dtype)
    

    # print()
    # turn into torch tensors
    image = torch.tensor(image, dtype=torch.float32)
    kspace = torch.tensor(kspace, dtype=torch.complex64)

    # print("image", image.shape, image.dtype)
    # print("kspace", kspace.shape, kspace.dtype)
    # print()
    
    # add batch dimension
    image = image.unsqueeze(0)
    kspace = kspace.unsqueeze(0)

    # print("image", image.shape, image.dtype)
    # print("kspace", kspace.shape, kspace.dtype)
    # print()

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image.numpy()[0, ..., 0])
    axs[0].set_title('Fully sampled knee MR image')
    axs[1].imshow(np.log(np.abs(kspace.numpy()[0, ..., 0])))
    axs[1].set_title('Fully sampled knee k-space in logscale')
    for i in [0, 1]:
        axs[i].axis('off')
    plt.savefig("gt_image_kspace.png")
    

    mask = gen_mask(kspace[0, ..., 0], accel_factor=4, seed=0)
    fourier_mask = np.repeat(mask.astype(np.float32), kspace.shape[1], axis=0)

    # print("mask", mask.shape, mask.dtype)
    # print("fourier_mask", fourier_mask.shape, fourier_mask.dtype)
    # print()

    plt.figure()
    plt.imshow(fourier_mask)
    plt.axis('off')
    plt.title('Mask simulating the undersampling in the Fourier space');
    plt.savefig("mask.png")
    
    fourier_mask = torch.from_numpy(fourier_mask[None, ..., None])
    # print("fourier_mask", fourier_mask.shape, fourier_mask.dtype)
    masked_kspace = fourier_mask * kspace

    # print("masked_kspace", masked_kspace.shape, masked_kspace.dtype)

    def crop_center(img, cropx, cropy=None):
        # taken from https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image/39382475
        if cropy is None:
            cropy = cropx
        y, x = img.shape[-2:]
        startx = x//2 - (cropx//2)
        starty = y//2 - (cropy//2)
        return img[..., starty:starty+cropy, startx:startx+cropx]


    # plt.figure()
    # seperate complex real and imaginary parts into a channel of its own
    

    result = torch.abs(torch.fft.fftshift(torch.fft.ifft2(masked_kspace[0, ..., 0], norm='ortho')))
    # print("result", result.shape, result.dtype)

    plt.figure()
    plt.imshow(crop_center(result.numpy(), 320))
    plt.axis('off')
    plt.title('Zero-filled reconstruction')
    plt.savefig("zero_filled_reconstruction.png")
    



    # plt.imshow(crop_center(np.abs(np.fft.fftshift(np.fft.ifft2(masked_kspace[..., 0], norm='ortho'))), 320))
    # print()

    
    # print("masked_kspace", masked_kspace.shape, masked_kspace.dtype)
    # print("fourier_mask", fourier_mask.shape, fourier_mask.dtype)

    # decompose the complex input into real and imaginary parts
    masked_kspace = torch.cat((masked_kspace.real, masked_kspace.imag), dim=-1)
    print("masked_kspace", masked_kspace.shape, masked_kspace.dtype)
    
    out = net(masked_kspace, fourier_mask)
    print(out.shape)

