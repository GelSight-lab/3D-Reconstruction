# -*- coding:utf-8 -*-
"""
Author: Yuxiang Ma
Date:   04/25/2022
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import open3d as o3d


class Visualizer:
    # data visualizers
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.n = np.size(X)

    def plot_gradients(self, gradx, grady, title='real data'):
        # title = 'real data' or 'ground truth'
        fig1, ax1 = plt.subplots()
        pc1 = plt.pcolormesh(self.X, self.Y, gradx, shading='auto')
        ax1.axis('equal')
        ax1.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
        fig1.colorbar(pc1)
        fig1.suptitle('Colormap of x-gradient {} '.format(title), fontsize=15)
        plt.show()

        fig2, ax2 = plt.subplots()
        pc2 = plt.pcolormesh(self.X, self.Y, grady, shading='auto')
        ax2.axis('equal')
        ax2.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
        fig2.colorbar(pc2)
        fig2.suptitle('Colormap of y-gradient {} '.format(title), fontsize=15)
        plt.show()

    def plot_geometry(self, r, title='distance'):
        # title = 'ground truth'
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        XX, YY = np.meshgrid(self.X, self.Y)
        ax.plot_surface(XX, YY, r, cmap=cm.coolwarm, rcount=201, ccount=201)
        fig.suptitle('3d mesh of {} on image plane'.format(title), fontsize=15)
        plt.show()

    def plot_colormap(self, r, title='distance', vmin=None, vmax=None):
        fig, ax = plt.subplots()
        if vmin is not None:
            pc = ax.pcolormesh(self.X, self.Y, r, shading='auto', vmin=vmin, vmax=vmax)
        else:
            pc = ax.pcolormesh(self.X, self.Y, r, shading='auto')
        ax.axis('equal')
        ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
        fig.colorbar(pc)
        fig.suptitle('Colormap of {} on image plane'.format(title), fontsize=15)
        plt.show()

# todo: 3d visualization with open3d
# # 3d visualization
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(pt)
    # o3d.visualization.draw_geometries([pcd1, o3d.geometry.TriangleMesh.create_coordinate_frame(1)], width=1280,
    #                                   height=720)
    #
    # # 3d visualization
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc_recon)
    # o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(1)], width=1280,
    #                                   height=720)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    #
    # pcd0 = o3d.geometry.PointCloud()
    # pcd0.points = o3d.utility.Vector3dVector(pt0)
    # vis.add_geometry(pcd0)
    # vis.update_geometry(pcd0)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.destroy_window()
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc_recon)
    # vis.add_geometry(pcd)