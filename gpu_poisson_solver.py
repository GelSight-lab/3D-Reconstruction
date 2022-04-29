# -*- coding:utf-8 -*-
"""
Author: Yuxiang Ma
Date:   04/04/2022
"""
import numpy as np
import jax.numpy as jnp
import scipy,scipy.fftpack
import math
import numba

# @numba.njit
def source_term(gradx, grady, dx, dy):
    # Laplacian
    gyy = (grady[1:, :-1] - grady[:-1, :-1])/dx
    gxx = (gradx[:-1, 1:] - gradx[:-1, :-1])/dy
    f = jnp.zeros(gradx.shape)

    f = f.at[:-1, 1:].add(gxx)
    f = f.at[1:, :-1].add(gyy)
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
    fw = jnp.fft.fftn(f, norm='ortho')
    # fsin = scipy.fftpack.dst(tt.T, norm='ortho', type=dst_type).T

    # Eigenvalues
    (x, y) = jnp.meshgrid(jnp.arange(1, f.shape[1] + 1), jnp.arange(1, f.shape[0] + 1), copy=True)
    # denom = (2 * jnp.cos(math.pi * x / (f.shape[1] + 2)) - 2)/dy**2 + (2 * jnp.cos(math.pi * y / (f.shape[0] + 2)) - 2)/dx**2
    denom = (2 * jnp.cos(math.pi * x / (f.shape[1] + 2)) - 2) / dy ** 2 + (
                2 * jnp.cos(math.pi * y / (f.shape[0] + 2)) - 2) / dx ** 2

    f = fw / denom

    # Inverse Discrete Sine Transform
    img_tt = jnp.fft.ifftn(f, norm='ortho')
    # img_tt = scipy.fftpack.idst(tt.T, norm='ortho', type=1).T

    # New center + old boundary
    result = boundary
    result = result.at[1:-1, 1:-1].add(img_tt)

    return result
