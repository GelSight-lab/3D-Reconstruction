# -*- coding:utf-8 -*-
"""
Author: Yuxiang Ma
Date:   04/20/2022
"""
# This script is an 3D reconstruction case of several balls on a sphere with simulated data
# Instead of using total source term (as in balls_on_sphere.py), poisson equation is solved with differential source term.
# Initial is
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import open3d as o3d

from fast_poisson_solver import poisson_solver, source_term
from data_generator import balls_on_sphere
from fisheye import Fisheye

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ###
    # initialize mesh on image plane and camera setting
    n = 251
    X = np.linspace(-1 / 2, 1 / 2, n)
    Y = np.linspace(-1 / 2, 1 / 2, n)
    dx = 1 / (n - 1)
    dy = 1 / (n - 1)

    fshy = Fisheye(X, Y, .8)

    ###
    # simu data with balls_on_sphere
    sphere = 10
    rtp = np.hstack((np.ones((X.size**2)).reshape(-1, 1)*sphere, fshy.tp))
    xyz = fshy.sph2cart(rtp)

    balls0 = np.array([[0, 0, 10, 5]])
    balls = np.array([[0, 0, 10, 5],
                      [0, 0.8, 14, 1.5],
                      [-1, -1, 13.5, 1],
                      [3, 0, 12.5, 0.8]])

    normals = xyz/sphere
    pt = xyz
    pt0, normals0 = balls_on_sphere(pt, normals, balls0)
    pt, normals = balls_on_sphere(pt, normals, balls)

    true0 = np.linalg.norm(pt0.reshape(n, n, 3), axis=2)
    true = np.linalg.norm(pt.reshape(n, n, 3), axis=2)  # ground truth

    ###
    # 3D reconstruction on image plane
    # compute right hand side
    grad0 = fshy.convert_norms(normals0)
    grad = fshy.convert_norms(normals)

    f0 = source_term(grad0[:, 0].reshape(n, n), grad0[:, 1].reshape(n, n), dx, dy)
    f = source_term(grad[:, 0].reshape(n, n), grad[:, 1].reshape(n, n), dx, dy)

    df = f - f0

    # set up boundary condition
    # todo: solve the displacement caused by boundary condition
    boundary = np.ones((n, n))*np.log(sphere)

    # solution is log(U) rather than U
    dlnU = poisson_solver(df, boundary, dx, dy)
    U = np.exp(dlnU + np.log(true0))

    pc_recon = fshy.sph2cart(np.hstack((U.reshape(-1, 1), fshy.tp)))
    ###
    # plots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    XX, YY = np.meshgrid(X, Y)
    ax.plot_surface(XX, YY, true, cmap=cm.coolwarm, rcount=201, ccount=201)
    # xyz scaled
    scale_x = X.max() - X.min()
    scale_y = Y.max() - X.min()
    scale_z = true.max() - true.min()
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    XX, YY = np.meshgrid(X, Y)
    ax.plot_surface(XX, YY, U, cmap=cm.coolwarm, rcount=201, ccount=201)
    # xyz scale
    scale_x = X.max() - X.min()
    scale_y = Y.max() - X.min()
    scale_z = U.max() - U.min()
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
    plt.show()

    # fig1 = plt.figure()
    # ax = fig1.add_subplot(111, projection='3d')
    # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    # # xyz scale
    # # scale_x = xyz[:,0].max() - xyz[:,0].min()
    # # scale_y = xyz[:,1].max() - xyz[:,1].min()
    # # scale_z = xyz[:,2].max() - xyz[:,2].min()
    # # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
    # plt.show()

    fig, ax = plt.subplots()
    pc = ax.pcolormesh(X, Y, U, shading='auto')
    ax.axis('equal')
    ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    fig.colorbar(pc)
    plt.show()

    fig, ax = plt.subplots()
    pc = ax.pcolormesh(X, Y, f, shading='auto')
    ax.axis('equal')
    ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    fig.colorbar(pc)
    plt.show()

    gradx = grad[:, 0].reshape(n, n)
    fig1, ax1 = plt.subplots()
    pc1 = plt.pcolormesh(X, Y, gradx, shading='auto')
    ax1.axis('equal')
    ax1.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    fig1.colorbar(pc1)
    plt.show()

    grady = grad[:, 1].reshape(n, n)
    fig2, ax2 = plt.subplots()
    pc2 = plt.pcolormesh(X, Y, grady, shading='auto')
    ax2.axis('equal')
    ax2.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    fig2.colorbar(pc2)
    plt.show()

    Gradx = gradx
    Grady = grady
    logtrue = np.log(true0)
    Gradx[1:-1, 1:-1] = (logtrue[1:-1, 2:] - logtrue[1:-1, :-2]) / 2 / dx
    Grady[1:-1, 1:-1] = (logtrue[2:, 1:-1] - logtrue[:-2, 1:-1]) / 2 / dy

    # fshy.compare_gradients(np.hstack((Gradx.reshape(-1, 1), Grady.reshape(-1, 1))))

    fig1, ax1 = plt.subplots()
    pc1 = plt.pcolormesh(X, Y, Gradx, shading='auto')
    ax1.axis('equal')
    ax1.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    fig1.colorbar(pc1)
    plt.show()

    grady = grad[:, 1].reshape(n, n)
    fig2, ax2 = plt.subplots()
    pc2 = plt.pcolormesh(X, Y, Grady, shading='auto')
    ax2.axis('equal')
    ax2.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    fig2.colorbar(pc2)
    plt.show()

    print(U.min())
    print(U.max())

    # 3d visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_recon)
    o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(1)])

    # # load data
    # pcfile = 'data/' + 'points.pkl'
    # normalfile = 'data/' + 'normals.pkl'
    # pc = pickle.load(open(pcfile, 'rb'))
    # normals = pickle.load(open(normalfile, 'rb'))
    # pc = np.asarray(pc)
    # normals = np.asarray(normals)
    #
    # truthfile = 'data/' + 'points_deformed.pkl'
    # truth = pickle.load(open(truthfile, 'rb'))
    # truth = np.asarray(truth)
    #
    # pc_recon = poisson_solver(pc, normals)
    #
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(pc_recon)
    # pcd1.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(10))
    # pcd1.orient_normals_towards_camera_location()
    # o3d.visualization.draw_geometries([pcd1, o3d.geometry.TriangleMesh.create_coordinate_frame(1)])
    #
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(truth)
    # pcd2.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(10))
    # pcd2.orient_normals_towards_camera_location()
    # o3d.visualization.draw_geometries([pcd2, o3d.geometry.TriangleMesh.create_coordinate_frame(2)])
    #
    # # generate image coordinates based on ground truth
    # rtp_truth = cart2sph(truth)
    # imagecoor = sph2fisheye(rtp_truth)
    #
    # grad_lnr = pretreat_normals(rtp_truth, normals)
    # rtp = poisson_solver(rtp_truth, grad_lnr)
