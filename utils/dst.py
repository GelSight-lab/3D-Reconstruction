# -*- coding:utf-8 -*-
"""
Author: Yuxiang Ma
Date:   05/05/2022

The algorithm was developed by applying Makhoul's fast cosine transform algorithm to sine transform. A blog (see below)
talking about discrete sine transform was consulted.

Ref: Makhoul, John. "A fast cosine transform in one and two dimensions." IEEE Transactions on Acoustics, Speech, and Signal Processing 28.1 (1980): 27-34.
     https://chasethedevil.github.io/post/discrete_sine_transform_fft/
"""
import numpy as np
import torch
import torch.nn as nn


def dst1(x):
    """
    Discrete Sine Transform, Type I
    :param x: the input signal
    :return: the DST-I of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[0]

    return torch.fft.rfft(torch.cat([torch.zeros(N, 1), x, torch.zeros(N, 1), -x.flip([1])], 1), dim=-1).imag[:, 1:-1].view(*x_shape)

def idst1(X):
    """
    The inverse of DST-I, which is just a scaled DST-I
    Our definition if idst1 is such that idst1(dst1(x)) == x
    :param X: the input signal
    :return: the inverse DST-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dst1(X) / (2 * (n + 1))


def dst(x, norm=None):
    """
    Discrete Sine Transform, Type II (a.k.a. the DST)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html
    The principle of this algorithm comes from Makhoul's dct algorithm, see:
    https://chasethedevil.github.io/post/discrete_sine_transform_fft/
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DST-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], -x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v, dim=-1)
    Vc = torch.roll(Vc, -1, -1)                       # offset the last dimension of Vc by -1
    k = - torch.arange(1, N+1, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = - Vc.real * W_i - Vc.imag * W_r

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(2)

    V = 2 * V.view(*x_shape)            # consistent with definition of scipy

    return V


def idst(X, norm=None):
    """
    The inverse to DST-II, which is a scaled Discrete Sine Transform, Type III
    Our definition of idst is that idst(dst(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html
    The principle of this algorithm comes from Makhoul's dct algorithm, see:
    https://chasethedevil.github.io/post/discrete_sine_transform_fft/
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DST-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(2)

    k = torch.arange(1, N+1, dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = torch.cat([-X_v.flip([1])[:, 1:], X_v[:, 0:1]*0], dim=1)
    V_t_i = X_v

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = - V_r - 1j*V_i
    V = torch.roll(V, 1, -1)

    v = torch.fft.ifft(V, dim=-1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += -v.flip([1])[:, :N // 2]

    x = x.real          # ignore imaginary part
    return x.view(*x_shape)


def dst_2d(x, norm=None):
    """
    2-dimentional Discrete Sine Transform, Type II (a.k.a. the DST)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DST-II of the signal over the last 2 dimensions
    """
    X1 = dst(x, norm=norm)
    X2 = dst(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idst_2d(X, norm=None):
    """
    The inverse to 2D DST-II, which is a scaled Discrete Sine Transform, Type III
    Our definition of idst is that idst_2d(dst_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DST-II of the signal over the last 2 dimensions
    """
    x1 = idst(X, norm=norm)
    x2 = idst(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dst_3d(x, norm=None):
    """
    3-dimentional Discrete Sine Transform, Type II (a.k.a. the DST)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DST-II of the signal over the last 3 dimensions
    """
    X1 = dst(x, norm=norm)
    X2 = dst(X1.transpose(-1, -2), norm=norm)
    X3 = dst(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idst_3d(X, norm=None):
    """
    The inverse to 3D DST-II, which is a scaled Discrete Sine Transform, Type III
    Our definition of idst is that idst_3d(dst_3d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dst.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DST-II of the signal over the last 3 dimensions
    """
    x1 = idst(X, norm=norm)
    x2 = idst(x1.transpose(-1, -2), norm=norm)
    x3 = idst(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


class LinearDST(nn.Linear):
    """Implement any DST as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DST matrix is stored, which will
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dst function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super().__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dst function
        I = torch.eye(self.N)
        if self.type == 'dst1':
            self.weight.data = dst1(I).data.t()
        elif self.type == 'idst1':
            self.weight.data = idst1(I).data.t()
        elif self.type == 'dst':
            self.weight.data = dst(I, norm=self.norm).data.t()
        elif self.type == 'idst':
            self.weight.data = idst(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDST layer to do a 2D DST.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)


def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDST layer to do a 3D DST.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)


if __name__ == '__main__':
    x = torch.Tensor(1000,4096)
    x.normal_(0,1)
    linear_dst = LinearDST(4096, 'dst')
    error = torch.abs(dst(x) - linear_dst(x))
    assert error.max() < 1e-3, (error, error.max())
    linear_idst = LinearDST(4096, 'idst')
    error = torch.abs(idst(x) - linear_idst(x))
    assert error.max() < 1e-3, (error, error.max())