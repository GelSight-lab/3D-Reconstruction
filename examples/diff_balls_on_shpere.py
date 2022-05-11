# -*- coding:utf-8 -*-
"""
Author: Yuxiang Ma
Date:   04/20/2022
"""
# This script is an 3D reconstruction case of several balls on a sphere with simulated data
# Instead of using total source term (as in balls_on_sphere.py), poisson equation is solved with differential source term.
# Initial is
import numpy as np
import open3d as o3d

from fast_poisson_solver import poisson_solver, source_term
from data_generator import place_balls, indent_balls
from fisheye import Fisheye
from Visualizer import Visualizer
import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ###
    # initialize mesh on image plane and camera setting
    start = time.time()
    n = 501
    X = np.linspace(-1 / 2, 1 / 2, n)
    Y = np.linspace(-1 / 2, 1 / 2, n)
    dx = 1 / (n - 1)
    dy = 1 / (n - 1)

    fshy = Fisheye(X, Y, 1)

    ###
    # simu data with balls_on_sphere
    sphere = 10
    rtp = np.hstack((np.ones((X.size**2)).reshape(-1, 1)*sphere, fshy.tp))
    xyz = fshy.sph2cart(rtp)

    balls0 = np.array([[0, 0, 8, 5]])
    # balls = np.array([[0, 0.8, 13,1.5],
    #                   [-1, -1, 13.5, 1],
    #                   [3, 0, 12.5, 0.8]])

    balls = np.array([[0, 0.8, 13.5, 1.5],
                      [-1, -1, 13, 1],
                      [3, 0, 12.5, 0.8]])

    normals = xyz/sphere
    pt = xyz
    pt, normals0 = place_balls(pt, normals, balls0)
    pt0 = pt.copy()

    pt, normals = indent_balls(pt, normals, balls)
    true = np.linalg.norm(pt.reshape(n, n, 3), axis=2)  # ground truth

    true0 = np.linalg.norm(pt0.reshape(n, n, 3), axis=2)

    time_sim = time.time()

    ###
    # 3D reconstruction on image plane
    # compute right hand side
    grad0 = fshy.convert_norms(normals0)
    grad  = fshy.convert_norms(normals)

    f0 = source_term(grad0[:, 0].reshape(n, n), grad0[:, 1].reshape(n, n), dx, dy)
    f = source_term(grad[:, 0].reshape(n, n), grad[:, 1].reshape(n, n), dx, dy)

    df = f - f0

    # set up boundary condition
    boundary = np.ones((n, n))*0

    # solution is log(U) rather than U
    dlnU = poisson_solver(df, boundary, dx, dy)

    # boundary = np.ones((n, n)) * np.log(sphere)
    # lnU0 = poisson_solver(f0, boundary, dx, dy)
    # U = np.exp(dlnU + lnU0)
    U = np.exp(dlnU) * true0
    time_solu = time.time()
    print("Elapsed time for reconstruction=%s"%(time_solu-time_sim))

    dlnU = poisson_solver(df, boundary, dx, dy)
    time_solu2 = time.time()
    U = np.exp(dlnU) * true0
    time_solu3 = time.time()
    print("Elapsed time for reconstruction=%s" % (time_solu2 - time_solu))
    print("Elapsed time for reconstruction=%s" % (time_solu3 - time_solu2))
    print(U.min())
    print(U.max())

    pc_recon = fshy.sph2cart(np.hstack((U.reshape(-1, 1), fshy.tp)))
    ###
    # plots
    vis = Visualizer(X, Y)

    vis.plot_geometry(U)
    vis.plot_geometry(true0, title='initial geometry')

    vis.plot_colormap(U, title='distance')
    vis.plot_colormap(f0, title='initial source')

    gradx = grad[:, 0].reshape(n, n)
    grady = grad[:, 1].reshape(n, n)
    vis.plot_gradients(gradx, grady, title='ground truth')

    Gradx = gradx
    Grady = grady
    logtrue = np.log(true)
    Gradx[1:-1, 1:-1] = (logtrue[1:-1, 2:] - logtrue[1:-1, :-2]) / 2 / dx
    Grady[1:-1, 1:-1] = (logtrue[2:, 1:-1] - logtrue[:-2, 1:-1]) / 2 / dy
    vis.plot_gradients(Gradx, Grady, title='ground truth')

    # fshy.compare_gradients(np.hstack((Gradx.reshape(-1, 1), Grady.reshape(-1, 1))))

    # 3d visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack((np.array([6, 1, 0]), pc_recon+np.array([6, 1, 0]))))
    o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(1)], width=1280,
                                      height=720)
