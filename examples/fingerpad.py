# -*- coding:utf-8 -*-
"""
Author: Yuxiang Ma
Date:   04/20/2022
"""
# This script is an 3D reconstruction case of several balls on a sphere with simulated data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import open3d as o3d
import pickle
import torch

from gpu_poisson_solver import poisson_solver, source_term
from fisheye import Fisheye

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ###
    # initialize mesh on image plane and camera setting
    n = int(640)
    m = int(480)

    calibfile = '../data/'+'calib.pkl'


    calib = pickle.load(open(calibfile, 'rb'))
    K = torch.as_tensor(calib.get('K'))
    D = torch.as_tensor(calib.get('D'))
    fshy = Fisheye(n, m, f=1/2, proj="stereographic")
    # fshy = Fisheye(n, m, K=K)

    dx = fshy.dx
    dy = fshy.dy
    X = fshy.x_img
    Y = fshy.y_img

    ###
    # load simu data of fingerpad touching a few cones
    file0 = '../data/' + 'image0.pkl'
    file1 = '../data/' + 'image1.pkl'

    # normals0 = torch.as_tensor(pickle.load(open(file0, 'rb')))
    # normals0 = torch.cat((normals0, torch.ones(m, n, 1)), dim=-1).detach()
    normals = torch.as_tensor(pickle.load(open(file1, 'rb')))
    fig1, ax1 = plt.subplots()
    pc1 = plt.imshow(normals[:,:,0])
    fig1.colorbar(pc1)
    plt.show()
    normals = torch.cat((normals, torch.ones(m, n, 1)), dim=-1).detach()

    ###
    # 3D reconstruction on image plane
    # compute right hand side
    # grad0 = fshy.convert_norms(normals0)
    grad = fshy.convert_norms(normals)
    # f0 = source_term(grad0[:, 0].reshape(m, n), grad0[:, 1].reshape(m, n), dx, dy)
    f1 = source_term(grad[:, 0].reshape(m, n), grad[:, 1].reshape(m, n), dx, dy)

    gradx = grad[:, 0].reshape(m, n)
    fig1, ax1 = plt.subplots()
    pc1 = plt.pcolormesh(X, Y, gradx, shading='auto')
    ax1.axis('equal')
    ax1.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    fig1.colorbar(pc1)
    plt.show()

    grady = grad[:, 1].reshape(m, n)
    fig2, ax2 = plt.subplots()
    pc2 = plt.pcolormesh(X, Y, grady, shading='auto')
    ax2.axis('equal')
    ax2.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    fig2.colorbar(pc2)
    plt.show()

    fig, ax = plt.subplots()
    pc = ax.pcolormesh(X, Y, f1, shading='auto')
    ax.axis('equal')
    ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    fig.colorbar(pc)
    plt.show()

    rt_img = fshy.rt_img.numpy()
    df = f1*10
    # set up boundary condition
    # todo: solve the displacement caused by boundary condition
    boundary = torch.ones((m, n))*2
    # boundary[1:-1, 1:-1] = 0

    # solution is log(U) rather than U
    lnU = poisson_solver(df, boundary, dx, dy)
    U = np.exp(lnU)

    pc_recon = fshy.sph2cart(torch.hstack((U.reshape(-1, 1), fshy.tp)))
    ###
    # plots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    XX, YY = torch.meshgrid(X, Y, indexing='xy')
    ax.plot_surface(XX.numpy(), YY.numpy(), lnU, cmap=cm.coolwarm, rcount=201, ccount=201)
    # xyz scale
    scale_x = X.max() - X.min()
    scale_y = Y.max() - X.min()
    scale_z = U.max() - U.min()
    # ax.get_proj = lambda: torch.dot(Axes3D.get_proj(ax), torch.diag([scale_x, scale_y, scale_z, 1]))
    plt.show()

    # fig1 = plt.figure()
    # ax = fig1.add_subplot(111, projection='3d')
    # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    # # xyz scale
    # # scale_x = xyz[:,0].max() - xyz[:,0].min()
    # # scale_y = xyz[:,1].max() - xyz[:,1].min()
    # # scale_z = xyz[:,2].max() - xyz[:,2].min()
    # # ax.get_proj = lambda: torch.dot(Axes3D.get_proj(ax), torch.diag([scale_x, scale_y, scale_z, 1]))
    # plt.show()

    fig, ax = plt.subplots()
    pc = ax.pcolormesh(X, Y, U, shading='auto')
    ax.axis('equal')
    ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    fig.colorbar(pc)
    plt.show()

    # Gradx = gradx
    # Grady = grady
    # logtrue = torch.log(true)
    # Gradx[1:-1, 1:-1] = (logtrue[1:-1, 2:] - logtrue[1:-1, :-2]) / 2 / dx
    # Grady[1:-1, 1:-1] = (logtrue[2:, 1:-1] - logtrue[:-2, 1:-1]) / 2 / dy

    # fshy.compare_gradients(torch.hstack((Gradx.reshape(-1, 1), Grady.reshape(-1, 1))))


    print(U.min())
    print(U.max())

    # 3d visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_recon)
    o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(1)])

    # load data
    # pcfile = 'data/' + 'points.pkl'
    # normalfile = 'data/' + 'normals.pkl'
    # pc = pickle.load(open(pcfile, 'rb'))
    # normals = pickle.load(open(normalfile, 'rb'))
    # pc = torch.as_tensor(pc)
    # normals = torch.as_tensor(normals)
    #
    # truthfile = 'data/' + 'points_deformed.pkl'
    # truth = pickle.load(open(truthfile, 'rb'))
    # truth = torch.as_tensor(truth)
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
