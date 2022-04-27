# -*- coding:utf-8 -*-
"""
Author: Yuxiang Ma
Date:   04/04/2022
"""
import optparse

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import cm
from fast_poisson_solver import poisson_solver, source_term
from sklearn.neighbors import KDTree
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D


def balls_on_flat(x, y, balls):
    # generate half balls on flat surface defined by X and Y
    # balls with a shape of (n, 3), each row defines a ball with location and radius
    X, Y = np.meshgrid(x, y)
    pt = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), np.zeros((X.size, 1))))
    normal = np.array([0, 0, 1])
    for ball in balls:
        kd = KDTree(pt)
        inds = kd.query_radius(ball[0:3].reshape(1, -1), ball[3])[0]
        dist = np.linalg.norm(pt[inds][:, 0:2] - ball[0:2], axis=1)     # dist on xy plane
        pt[inds, 2] = ball[2]-np.sqrt(ball[3] ** 2 - dist.reshape(-1,) ** 2)
        # dist = np.linalg.norm(pt[inds] - ball[0:3], axis=1)           # dist on 3d real world
        # pt[inds]
        # -= np.sqrt(ball[3]**2 + ball[2]**2 - dist.reshape(-1, 1)**2) * normal

    # pt[:,2] = signal.convolve2d(pt[:, 2].reshape(x.shape[0], y.shape[0]), np.ones((3, 3)), mode='same').reshape((-1,))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(10))
    pcd.orient_normals_towards_camera_location()

    return np.asarray(pcd.points), np.asarray(pcd.normals)


if __name__ == "__main__":
    n = 501
    X = np.linspace(-1/2, 1/2, n)
    Y = np.linspace(-1/2, 1/2, n)
    dx = 1/(n-1)
    dy = 1/(n-1)

    src_loc = int(n/2)
    f = np.zeros((n, n))
    f[src_loc-1:src_loc+2, src_loc-1:src_loc+2] = 1

    # balls = np.array([[0, 0, 0.1, 0.4]])
    balls = np.array([[0, 0, 0.3, 0.5], [0, 0, -0.1, 0.2]])
    pt, normals = balls_on_flat(X, Y, balls)
    gradx = -normals[:, 0]/normals[:, 2]
    grady = -normals[:, 1]/normals[:, 2]
    inf_num = 1.5

    f = source_term(gradx.reshape(n, n), grady.reshape(n, n), dx, dy)

    # fig, ax = plt.subplots()
    # ax.pcolormesh(X, Y, -pt[:, 2].reshape(n, n), shading='auto')
    # plt.show()

    boundary = np.zeros((n, n))

    U = poisson_solver(f, boundary, dx, dy)
    # data plot
    # plt.style.use('_mpl-gallery-nogrid')
    true = -pt[:,2].reshape(n, n)
    # fig, ax = plt.subplots()
    # ax.pcolormesh(X, Y, -pt[:,2].reshape(n,n), shading='auto')
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    XX, YY = np.meshgrid(X, Y)
    ax.plot_surface(XX, YY, true, cmap=cm.coolwarm, rcount=201, ccount=201)
    # xyz scaled
    scale_x = X.max() - X.min()
    scale_y = Y.max() - X.min()
    scale_z = U.max() - U.min()
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    XX, YY = np.meshgrid(X, Y)
    ax.plot_surface(XX, YY, -U, cmap=cm.coolwarm, rcount=201, ccount=201)
    # xyz scale
    scale_x = X.max()-X.min()
    scale_y = Y.max()-X.min()
    scale_z = U.max()-U.min()
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
    plt.show()

    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, -U, shading='auto')
    ax.axis('equal')
    ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    fig, ax = plt.subplots()
    ax.pcolormesh(X, Y, f, shading='auto')
    ax.axis('equal')
    ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    plt.show()

    print(U.min())







