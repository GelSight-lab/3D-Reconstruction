# -*- coding:utf-8 -*-
"""
Author: Yuxiang Ma
Date:   05/05/2022
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
    N = x_shape[-1]
    # x = torch.cat([torch.zeros(1, N), x.view(-1, N)])

    # return torch.rfft(torch.cat([x, -x.flip([1])[:, 1:-1]], dim=1), 1).imag[:, :, 0].view(*x_shape)/2
    return torch.fft.rfft(torch.cat([torch.zeros(N, 1), x, torch.zeros(N, 1), -x.flip([1])], 1), 1).imag.view(*x_shape) / 2

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
    Discrete Sine Transform, Type I (a.k.a. the DST)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dst.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DST-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idst(X, norm=None):
    """
    The inverse to DST-II, which is a scaled Discrete Sine Transform, Type III
    Our definition of idst is that idst(dst(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dst.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DST-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.irfft(V, 1, onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dst_2d(x, norm=None):
    """
    2-dimentional Discrete Sine Transform, Type II (a.k.a. the DST)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dst.html
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
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dst.html
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
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dst.html
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
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dst.html
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