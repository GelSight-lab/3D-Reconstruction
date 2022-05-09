# -*- coding:utf-8 -*-
"""
Author: Yuxiang Ma
Date:   04/04/2022
"""
import numpy as np

import torch
import scipy,scipy.fftpack
import math
from dst import dst1, idst1

def source_term(gradx, grady, dx, dy):
    # Laplacian
    gyy = (grady[1:, :-1] - grady[:-1, :-1])/dx
    gxx = (gradx[:-1, 1:] - gradx[:-1, :-1])/dy
    f = torch.zeros(gradx.shape)

    f[:-1, 1:] += gxx
    f[1:, :-1] += gyy
    return f

# def diff_source():

def poisson_solver(f, boundary, dx, dy):
    # Thanks to Dr. Ramesh Raskar for providing the original matlab code from which this is derived
    # Dr. Raskar's version is available here: http://web.media.mit.edu/~raskar/photo/code.pdf

    # Boundary image
    # boundary[1:-1, 1:-1] = 0

    # Subtract boundary contribution
    f_bp = -4*boundary[1:-1, 1:-1] + boundary[1:-1, 2:] + boundary[1:-1, 0:-2] + boundary[2:,1:-1] + boundary[0:-2,1:-1]
    f = f[1:-1, 1:-1] - f_bp/dx**2

    # Discrete Sine Transform
    dst_type = 1
    ftt = dst1(f)
    fw = dst1(ftt.T).T
    # fw = torch.fft.fft2(f, norm='ortho')
    # fsin = scipy.fftpack.dst(tt.T, norm='ortho', type=dst_type).T

    # Eigenvalues
    x = torch.arange(1, f.shape[1] + 1)
    y = torch.arange(1, f.shape[0] + 1)
    (x, y) = torch.meshgrid(x, y, copy=True)

    # denom = (2 * torch.cos(math.pi * x / (f.shape[1] + 2)) - 2)/dy**2 + (2 * torch.cos(math.pi * y / (f.shape[0] + 2)) - 2)/dx**2
    denom = (2 * torch.cos(math.pi * x / (f.shape[1] + 2)) - 2) / dy ** 2 + (
                2 * torch.cos(math.pi * y / (f.shape[0] + 2)) - 2) / dx ** 2

    f = fw / denom
    # Inverse Discrete Sine Transform
    tt = idst1(f)
    img_tt = idst1(tt.T).T
    # img_tt = torch.fft.ifft2(f, norm='ortho')

    # New center + old boundary
    result = boundary
    result = result.at[1:-1, 1:-1].add(img_tt)

    return result
