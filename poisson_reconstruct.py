# import open3d as o3d
import numpy as np
# from sklearn.neighbors import KDTree
import pickle
from fast_poisson_solver import poisson_solver, source_term
from data_generator import balls_on_sphere
import matplotlib.pyplot as plt
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D
# This script conducts 3D reconstruction on any curved surface with a fisheye camera
# by solving poisson equation on the image plane with the natural mesh.

class Fisheye:
    # spherical coordinate(rtp) to image coordinate(imagecoor)
    #
    # proj refers to the type of projection transformation(https://wiki.panotools.org/Fisheye_Projection)
    # equidistant(default):      R=f*theta
    # stereographic:    R=2*f*tan(theta/2)
    # orthographic:     R=f*sin(theta)
    # equisolid:        R=2*f*sin(theta/2)
    # thoby, PTGui...
    projs = ["equidistant", "stereographic", "orthographic", "equisolid"]

    def __init__(self, x_img, y_img, f, proj="equidistant"):
        self.x_img = x_img
        self.y_img = y_img
        self.xy_img = np.hstack((
            np.tile(x_img, y_img.size).reshape(-1, 1),
            np.repeat(y_img, x_img.size).reshape(-1, 1) ))  # (x, y) of all points and iterate y first
        self.proj = proj
        self.f = f
        self.p = self.projs.index(proj)

        self.cart2polar()
        self.polar2sph()

    def cart2polar(self):
        # ! On image plane, converting cartesian coordinates to polar coordinates.
        self.rt_img = np.zeros(self.xy_img.shape)
        self.rt_img[:, 0] = np.sqrt(self.xy_img[:, 0] ** 2 + self.xy_img[:, 1] ** 2)
        self.rt_img[:, 1] = np.arctan2(self.xy_img[:, 1], self.xy_img[:, 0])

    def polar2cart(self):
        # ! On image plane, converting polar coordinates to cartesian coordinates.
        self.xy_img = np.zeros(self.rt_img.shape)
        self.xy_img[:, 0] = self.rt_img[:, 0] * np.cos(self.rt_img[:, 1])
        self.xy_img[:, 1] = self.rt_img[:, 0] * np.sin(self.rt_img[:, 1])

    def polar2sph(self):
        # ! Converting polar coordinates to spherical coordinates (without r)
        self.tp = self.rt_img
        if self.p==0: # equidistant
                self.tp[:, 0] = self.f * self.tp[:, 0]
        elif self.p==1:  # stereographic
                self.tp[:, 0] = 2 * self.f * np.tan(self.tp[:, 0] / 2)
        elif self.p==2:  # orthographic
                self.tp[:, 0] = self.f * np.sin(self.tp[:, 0])
        elif self.p==3:  # equisolid
                self.tp[:, 0] = 2 * self.f * np.sin(self.tp[:, 0] / 2)
        else:  # not defined
                print('Projection "' + self.proj + '" was not defined.')
                exit(-1)

    def convert_norms(self, normals):
        # ! converting norms in real space to gradients(norms) in computational space (x_img, y_img, r)
        # 1, transform normals to spherical coordinates
        # 2, turn normals in spherical coordinates into gradient of ln(r) with regard to (rho, phi) on image plane
        # 3, transform gradient of ln(r) from polar to cartesian (on image plane)

        # transform normals to spherical coordinates
        # local spherical frameworks (sphframes shape 3*3*n)
        sph_frames = np.dstack(
            (np.vstack(
                (np.sin(self.tp[:, 0]) * np.cos(self.tp[:, 1]), np.sin(self.tp[:, 0]) * np.sin(self.tp[:, 1]),
                 np.cos(self.tp[:, 0]))).T,
             np.vstack(
                 (np.cos(self.tp[:, 0]) * np.cos(self.tp[:, 1]), np.cos(self.tp[:, 0]) * np.sin(self.tp[:, 1]),
                  -np.sin(self.tp[:, 0]))).T,
             np.vstack((-np.sin(self.tp[:, 1]), np.cos(self.tp[:, 1]), np.zeros((self.tp.shape[0])))).T))
        norm_sph = np.sum((sph_frames.reshape(-1, 3) * normals.reshape(-1, 1)).reshape(-1, 3, 3), axis=1)

        # turn norms into gradient with respect to (rho, phi) on image plane
        grad_lnr = - norm_sph[:, 1:] / norm_sph[:, 0].reshape(-1, 1)    # normalization
        grad_lnr[:, 1] = grad_lnr[:, 1] * np.cos(self.tp[:, 0])          # derivatives with regard to (theta, phi)

        self.grad_lnr = grad_lnr

        grad_lnr = self.gradient2image(grad_lnr)                        # transform to image plane

        # On image plane, transform gradient of ln(r) from polar to cartesian
        polar_frames = np.dstack(
            (np.vstack(
                (np.cos(self.rt_img[:, 1]), -np.sin(self.rt_img[:, 1]))).T,
             np.vstack((np.sin(self.rt_img[:, 1]), np.cos(self.rt_img[:, 1]))).T))

        grad = np.sum((polar_frames.reshape(-1, 2)*grad_lnr.reshape(-1, 1)).reshape(-1, 2, 2), axis=1)  # coordinate transformation

        return grad

    def compare_gradients(self, Grad):
        polar_frames = np.dstack(
            (np.vstack(
                (np.cos(self.rt_img[:, 1]), np.sin(self.rt_img[:, 1]))).T,
             np.vstack((-np.sin(self.rt_img[:, 1]), np.cos(self.rt_img[:, 1]))).T))

        Grad_lnr = np.sum((polar_frames.reshape(-1, 2) * Grad.reshape(-1, 1)).reshape(-1, 2, 2), axis=1)  # coordinate transformation

        Grad_lnr[:, 0] = Grad_lnr[:, 0] * self.f

        Gradt = Grad_lnr[:, 0].reshape(n, n)
        Gradp = Grad_lnr[:, 1].reshape(n, n)
        gradt = self.grad_lnr[:, 0].reshape(n, n)
        gradp = self.grad_lnr[:, 1].reshape(n, n)

        fig1, ax1 = plt.subplots()
        pc1 = plt.pcolormesh(Gradt, shading='auto')
        ax1.axis('equal')
        fig1.colorbar(pc1)
        plt.show()

        fig2, ax2 = plt.subplots()
        pc2 = plt.pcolormesh(Gradp, shading='auto')
        ax2.axis('equal')
        fig2.colorbar(pc2)
        plt.show()

        fig1, ax1 = plt.subplots()
        pc1 = plt.pcolormesh(gradt, shading='auto')
        ax1.axis('equal')
        fig1.colorbar(pc1)
        plt.show()

        fig2, ax2 = plt.subplots()
        pc1 = plt.pcolormesh(gradp, shading='auto')
        ax2.axis('equal')
        fig2.colorbar(pc1)
        plt.show()

        fig1, ax1 = plt.subplots()
        pc1 = plt.pcolormesh(gradt-Gradt, shading='auto')
        ax1.axis('equal')
        fig1.colorbar(pc1)
        plt.show()

        fig2, ax2 = plt.subplots()
        pc1 = plt.pcolormesh(gradp-Gradp, shading='auto')
        ax2.axis('equal')
        fig2.colorbar(pc1)
        plt.show()

        return Grad_lnr

    def gradient2image(self, grad):

        if self.p==0:  # equidistant
                grad[:, 0] = grad[:, 0] / self.f
        elif self.p==1:  # stereographic
                grad[:, 0] = grad[:, 0] / self.f / (1 + (self.rt_img[:, 0] / 2 * self.f) ** 2)
        elif self.p==2:  # orthographic
                grad[:, 0] = grad[:, 0] / self.f / np.sqrt(1 - (self.rt_img[:, 0] * self.f) ** 2)
        elif self.p==3:  # equisolid
                grad[:, 0] = grad[:, 0] / self.f / np.sqrt(1 - (self.rt_img[:, 0] / 2 / self.f) ** 2)
        else:  # not defined
                print('Projection "' + self.proj + '" was not defined.')
                exit(-1)
        return grad


def cart2sph(xyz):
    # cartesian coordinate(xyz) to spherical coordinate(rtp)
    x2y2 = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    rtp = np.zeros(xyz.shape)
    rtp[:, 0] = np.sqrt(x2y2 + xyz[:, 2] ** 2)
    rtp[:, 1] = np.arctan2(np.sqrt(x2y2), xyz[:, 2])
    rtp[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return rtp


def sph2cart(rtp):
    # spherical coordinate(rtp) to cartesian coordinate(xyz)
    xyz = np.zeros(rtp.shape)
    xyz[:, 2] = rtp[:, 0] * np.cos(rtp[:, 1])
    r = rtp[:, 0] * np.sin(rtp[:, 1])
    xyz[:, 1] = r * np.sin(rtp[:, 2])
    xyz[:, 0] = r * np.cos(rtp[:, 2])
    return xyz


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # mesh on image plane
    n = 201
    X = np.linspace(-1 / 2, 1 / 2, n)
    Y = np.linspace(-1 / 2, 1 / 2, n)
    dx = 1 / (n - 1)
    dy = 1 / (n - 1)
    focal = 1

    sphere = 10
    fshy = Fisheye(X, Y, focal)
    rtp = np.hstack((np.ones((X.size**2)).reshape(-1, 1)*sphere, fshy.tp))
    xyz = sph2cart(rtp)
    # generate fake data
    # balls = np.array([[0, 0, 9, 4.5],
    #                   [1, 1, 13, 2]])
    balls = np.array([[1, 1, 9, 3]])
    normals = xyz/sphere
    pt = xyz
    pt, normals = balls_on_sphere(pt, normals, balls)

    # solve on image plane
    grad = fshy.convert_norms(normals)

    f = source_term(grad[:, 0].reshape(n, n), grad[:, 1].reshape(n, n), dx, dy)

    # todo: solve the displacement caused by boundary condition
    boundary = np.ones((n, n))*np.log(sphere)
    # boundary[1:-1, 1:-1] = 0

    lnU = poisson_solver(f, boundary, dx, dy)
    U = np.exp(lnU)

    true = np.linalg.norm(pt.reshape(n, n, 3), axis=2)

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

    # fig, ax = plt.subplots()
    # pc = ax.pcolormesh(X, Y, -U, shading='auto')
    # ax.axis('equal')
    # ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    # fig.colorbar(pc)
    # plt.show()

    # fig, ax = plt.subplots()
    # pc = ax.pcolormesh(X, Y, f, shading='auto')
    # ax.axis('equal')
    # ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    # fig.colorbar(pc)
    # plt.show()
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

    # true = np.log(true)
    # Gradx[1:-1, 1:-1] = (true[1:-1, 2:] - true[1:-1, :-2])/2/dx
    # Grady[1:-1, 1:-1] = (true[2:, 1:-1] - true[:-2, 1:-1])/2/dy
    #
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



    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
    # xyz scale
    # scale_x = xyz[:,0].max() - xyz[:,0].min()
    # scale_y = xyz[:,1].max() - xyz[:,1].min()
    # scale_z = xyz[:,2].max() - xyz[:,2].min()
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1]))
    plt.show()

    print(U.min())
    print(U.max())


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
