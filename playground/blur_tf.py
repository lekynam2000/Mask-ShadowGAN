import os
import sys
import time
from functools import reduce

import numpy as np
import scipy.stats as st
import tensorflow as tf
from PIL import Image
from numpy import asarray


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.double)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


def blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')

if __name__=="__main__":
    img_path = "../dataset/ISTD_Dataset/test/test_C/100-5.png"
    img = asarray(Image.open(img_path).convert('RGB')).astype(np.double)
    img = np.expand_dims(img , axis=0)
    img = blur(img).numpy()
    print(img)
    print(img.shape)
    # print(type(img))
    # img = Image.fromarray(img)
    # img.save("playground/100-5_blur.png")

