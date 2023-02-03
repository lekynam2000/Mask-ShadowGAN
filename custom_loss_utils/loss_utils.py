import os
import sys
import time
from functools import reduce

import torch
from torch import nn
import numpy as np
import scipy.stats as st
from PIL import Image
from numpy import asarray

#For computing color loss
def gauss_filter(kernlen=21, nsig=3, channels=3):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    kernel = torch.from_numpy(out_filter)
    kernel = kernel.permute(3,2,1,0)
    kernel = kernel.repeat(channels,1,1,1)
    print(f"Kernel shape: {kernel.shape}")
    filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                 kernel_size=kernlen, groups=channels, padding='same', bias=False)
    filter.weight.data = kernel
    filter = filter.double()
    print(filter.weight.data)
    return filter

def blur(x):
    filter = gauss_filter(21,3,3)
    return filter(x)

#For computing style loss
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

if __name__=="__main__":
    img_path = "../dataset/ISTD_Dataset/test/test_C/100-5.png"
    img = asarray(Image.open(img_path).convert('RGB')).astype(np.double)
    print(img)
    img = np.expand_dims(img , axis=0)
    img = torch.from_numpy(img)
    img = img.permute(0,3,1,2).double()
    img = blur(img)
    img=img.permute(0,2,3,1)
    img=img.detach().numpy()
    print(img)
    print(img.shape)