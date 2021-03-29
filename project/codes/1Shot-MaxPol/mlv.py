import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm

PAD_SIZE = 1
FILTER_SIZE = 3


def calculate_MLV(image, pad_size=PAD_SIZE, filter_size=FILTER_SIZE, rgb=True, return_weight_matrix=False, weighted=True):
    assert filter_size % 2 == 1, "Filter must be of odd shape"
    N = filter_size//2
    if rgb:
        image = np.asfarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    ax1, ax2 = np.shape(image)
    image_ = np.zeros((ax1+2*pad_size, ax2+2*pad_size))
    image_[pad_size:-pad_size, pad_size:-pad_size] = image
    image = image_
    phi = np.zeros((ax1, ax2), dtype=np.float)
    for i in range(N, ax1+N):
        for j in range(N, ax2+N):
            image_patch = image[i-N:i+N+1, j-N:j+N+1]
            phi[i-N, j-N] = np.max(np.abs(image_patch-image[i, j]))

    phi /= 255
    if weighted:
        rank_matrix = phi.ravel().argsort().argsort().reshape(phi.shape)/(np.size(phi)-1)
        phi_w = np.exp(rank_matrix)*phi
        sigma = np.std(phi_w)
    else:
        sigma = np.std(phi)

    if not return_weight_matrix:
        return sigma
    else:
        if weighted:
            return sigma, phi_w
        else:
            return sigma, phi


basedir = "../Data/McM"
file_names = os.listdir(basedir)
sigma_vals = []
for file_name in tqdm(file_names):
    image = cv2.imread(os.path.join(basedir, file_name))
    sigma_vals.append(calculate_MLV(image))

print(sigma_vals)
