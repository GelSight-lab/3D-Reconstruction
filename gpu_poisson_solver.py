# -*- coding:utf-8 -*-
"""
Author: Yuxiang Ma
Date:   04/26/2022
"""
# -*- coding:utf-8 -*-
"""
Author: Yuxiang Ma
Date:   04/04/2022
"""
# import cp
# import scipy, scipy.fftpack
import math
import cupy as cp

def source_term(gradx, grady, dx, dy):
    # Laplacian
    gyy = (grady[1:, :-1] - grady[:-1, :-1])/dx
    gxx = (gradx[:-1, 1:] - gradx[:-1, :-1])/dy
    f = cp.zeros(gradx.shape)

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
    tt = cp.scipy.fftpack.dst(f, norm='ortho', type=dst_type)
    fsin = cp.scipy.fftpack.dst(tt.T, norm='ortho', type=dst_type).T

    # Eigenvalues
    (x, y) = cp.meshgrid(range(1, f.shape[1] + 1), range(1, f.shape[0] + 1), copy=True)
    # denom = (2 * cp.cos(math.pi * x / (f.shape[1] + 2)) - 2)/dy**2 + (2 * cp.cos(math.pi * y / (f.shape[0] + 2)) - 2)/dx**2
    denom = (2 * cp.cos(math.pi * x / (f.shape[1] + 2)) - 2) / dy ** 2 + (
                2 * cp.cos(math.pi * y / (f.shape[0] + 2)) - 2) / dx ** 2

    f = fsin / denom

    # Inverse Discrete Sine Transform
    tt = cp.scipy.fftpack.idst(f, norm='ortho', type=1)
    img_tt = cp.scipy.fftpack.idst(tt.T, norm='ortho', type=1).T

    # New center + old boundary
    result = boundary
    result[1:-1, 1:-1] += img_tt

    return result