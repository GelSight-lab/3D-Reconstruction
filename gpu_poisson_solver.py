# -*- coding:utf-8 -*-
"""
Author: Yuxiang Ma
Date:   04/04/2022
"""

import torch
import math
from utils.dst import dst_2d, idst_2d

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

    # Subtract boundary contribution
    f_bp = -4*boundary[1:-1, 1:-1] + boundary[1:-1, 2:] + boundary[1:-1, 0:-2] + boundary[2:,1:-1] + boundary[0:-2,1:-1]
    f = f[1:-1, 1:-1] - f_bp/dx**2

    # Discrete Sine Transform
    fsin = dst_2d(f)

    # Eigenvalues
    x = torch.arange(1, f.shape[1] + 1)
    y = torch.arange(1, f.shape[0] + 1)
    (x, y) = torch.meshgrid(x, y, indexing='xy')

    denom = (2 * torch.cos(math.pi * x / (f.shape[1] + 1)) - 2)/dy**2 + (2 * torch.cos(math.pi * y / (f.shape[0] + 1)) - 2)/dx**2
    # denom = (2 * torch.cos(math.pi * x / (f.shape[1] + 2)) - 2) / dy ** 2 + (
    #             2 * torch.cos(math.pi * y / (f.shape[0] + 2)) - 2) / dx ** 2

    f = fsin / denom
    # Inverse Discrete Sine Transform
    img_tt = idst_2d(f)

    # New center + old boundary
    result = boundary
    result[1:-1, 1:-1] += img_tt

    return result
