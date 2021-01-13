# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:43:15 2020

@author: Pierre
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def vec3(arr):
    return np.repeat(arr.reshape(512, 512, 1), 3, axis=2)

def grad2D(f):
    gradX = f - f[np.mod(np.arange(512)+1,512.0).astype(int)]
    gradY = f - f[:, np.mod(np.arange(512)+1,512.0).astype(int)]
    return [gradX, gradY]
            

noise = gaussian_filter(np.power(plt.imread("noises.png")[:,:,2].astype("float"), 1.0),1.0, mode="wrap")
plt.imshow(noise[np.mod(np.arange(0,512,16),512.0).astype(int)])
grad = np.gradient(noise*2.0)

n = np.zeros([512,512,3])
n[:,:,0] = grad[0]
n[:,:,1] = grad[1]
n[:,:,2] = 1.0
n /= vec3(np.linalg.norm(n,axis=2))

txt1 ="vec3(" + str(n[:,:,0].min()) + "," + str(n[:,:,1].min()) + "," + str(n[:,:,2].min()) + ");"
n[:,:,0] -= n[:,:,0].min()
n[:,:,1] -= n[:,:,1].min()
n[:,:,2] -= n[:,:,2].min()

print("wave = wave * vec3(", n[:,:,0].max(), "," ,n[:,:,1].max(), "," , n[:,:,2].max(), ") + " + txt1, sep="")
n[:,:,0] /= n[:,:,0].max()
n[:,:,1] /= n[:,:,1].max()
n[:,:,2] /= n[:,:,2].max()
print(n[:,:,0].min(),n[:,:,1].min(), n[:,:,2].min())
print(n[:,:,0].max(),n[:,:,1].max(), n[:,:,2].max())

nH = np.zeros([512,512,4])
nH[:,:,:3] = n
nH[:,:,3] = noise

plt.imsave("wave.png", nH)