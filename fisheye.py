"""
Fisheye class defines transformations of coordinates and normals
Author: Yuxiang Ma
Date:   04/19/2022
"""
# import numpy as np
import jax.numpy as jnp

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
        x, y = jnp.meshgrid(x_img, y_img)
        self.xy_img = jnp.hstack((x.reshape(-1, 1), y.reshape(-1, 1) ))  # (x, y) of all points and iterate y first
        self.proj = proj
        self.f = f
        self.p = self.projs.index(proj)

        self.cart2polar()
        self.polar2sph()

    def cart2polar(self):
        # ! On image plane, converting cartesian coordinates to polar coordinates.
        r_img = jnp.sqrt(self.xy_img[:, 0] ** 2 + self.xy_img[:, 1] ** 2)
        t_img = jnp.arctan2(self.xy_img[:, 1], self.xy_img[:, 0])
        self.rt_img = jnp.hstack((r_img.reshape(-1, 1), t_img.reshape(-1, 1)))

    def polar2cart(self):
        # ! On image plane, converting polar coordinates to cartesian coordinates.
        x_img = self.rt_img[:, 0] * jnp.cos(self.rt_img[:, 1])
        y_img = self.rt_img[:, 0] * jnp.sin(self.rt_img[:, 1])
        self.xy_img = jnp.hstack((x_img.reshape(-1, 1), y_img.reshape(-1, 1)))

    def polar2sph(self):
        # ! Converting polar coordinates to spherical coordinates (without r)
        if self.p==0:       # equidistant
                self.tp = self.rt_img.at[:, 0].set(self.rt_img[:, 0] / self.f)
        elif self.p==1:     # stereographic
                self.tp = self.rt_img.at[:, 0].set(jnp.arctan(self.rt_img[:, 0] / 2 / self.f))
        elif self.p==2:     # orthographic
                self.tp = self.rt_img.at[:, 0].set(jnp.arcsin(self.rt_img[:, 0] / self.f))
        elif self.p==3:     # equisolid
                self.tp = self.rt_img.at[:, 0].set(2 * jnp.arcsin(self.rt_img[:, 0] / 2 / self.f))
        else:               # not defined
                print('Projection "' + self.proj + '" was not defined.')
                exit(-1)

    def sph2polar(self):
        # ! Converting polar coordinates to spherical coordinates (without r)
        if self.p==0:       # equidistant
                self.rt_img = self.tp.at[:, 0].set(self.f * self.tp[:, 0])
        elif self.p==1:     # stereographic
                self.rt_img = self.tp.at[:, 0].set(2 * self.f * jnp.tan(self.tp[:, 0] / 2))
        elif self.p==2:     # orthographic
                self.rt_img = self.tp.at[:, 0].set(self.f * jnp.sin(self.tp[:, 0]))
        elif self.p==3:     # equisolid
                self.rt_img = self.tp.at[:, 0].set(2 * self.f * jnp.sin(self.tp[:, 0] / 2))
        else:               # not defined
                print('Projection "' + self.proj + '" was not defined.')
                exit(-1)

    # these two functions are projections between cartesian coordinate and spherical coordinate in real world
    def cart2sph(self, xyz):
        # cartesian coordinate(xyz) to spherical coordinate(rtp)
        x2y2 = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        r = jnp.sqrt(x2y2 + xyz[:, 2] ** 2)
        t = jnp.arctan2(jnp.sqrt(x2y2), xyz[:, 2])
        p = jnp.arctan2(xyz[:, 1], xyz[:, 0])
        return jnp.hstack((r.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1)))

    def sph2cart(self, rtp):
        # spherical coordinate(rtp) to cartesian coordinate(xyz)
        z = rtp[:, 0] * jnp.cos(rtp[:, 1])
        r = rtp[:, 0] * jnp.sin(rtp[:, 1])
        y = r * jnp.sin(rtp[:, 2])
        x = r * jnp.cos(rtp[:, 2])
        return jnp.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))

    def convert_norms(self, normals):
        # ! converting norms in real space to gradients(norms) in computational space (x_img, y_img, r)
        # 1, transform normals to spherical coordinates
        # 2, turn normals in spherical coordinates into gradient of ln(r) with regard to (rho, phi) on image plane
        # 3, transform gradient of ln(r) from polar to cartesian (on image plane)

        # transform normals to spherical coordinates
        # local spherical frameworks (sphframes shape 3*3*n)
        sph_frames = jnp.dstack(
            (jnp.vstack(
                (jnp.sin(self.tp[:, 0]) * jnp.cos(self.tp[:, 1]), jnp.sin(self.tp[:, 0]) * jnp.sin(self.tp[:, 1]),
                 jnp.cos(self.tp[:, 0]))).T,
             jnp.vstack(
                 (jnp.cos(self.tp[:, 0]) * jnp.cos(self.tp[:, 1]), jnp.cos(self.tp[:, 0]) * jnp.sin(self.tp[:, 1]),
                  -jnp.sin(self.tp[:, 0]))).T,
             jnp.vstack((-jnp.sin(self.tp[:, 1]), jnp.cos(self.tp[:, 1]), jnp.zeros((self.tp.shape[0])))).T))
        norm_sph = jnp.sum((sph_frames.reshape(-1, 3) * normals.reshape(-1, 1)).reshape(-1, 3, 3), axis=1)

        # turn norms into gradient with respect to (rho, phi) on image plane
        grad_lnr = - norm_sph[:, 1:] / norm_sph[:, 0].reshape(-1, 1)    # normalization
        grad_lnr = grad_lnr.at[:, 1].set(grad_lnr[:, 1] * jnp.cos(self.tp[:, 0]))           # derivatives with regard to (theta, phi)
        self.grad_lnr = grad_lnr

        grad_lnr = self.gradient2image(grad_lnr)                        # transform to image plane

        # On image plane, transform gradient of ln(r) from polar to cartesian
        polar_frames = jnp.dstack(
            (jnp.vstack(
                (jnp.cos(self.rt_img[:, 1]), -jnp.sin(self.rt_img[:, 1]))).T,
             jnp.vstack((jnp.sin(self.rt_img[:, 1]), jnp.cos(self.rt_img[:, 1]))).T))

        grad = jnp.sum((polar_frames.reshape(-1, 2)*grad_lnr.reshape(-1, 1)).reshape(-1, 2, 2), axis=1)  # coordinate transformation

        return grad

    def gradient2image(self, grad):
        if self.p==0:  # equidistant
                grad = grad.at[:, 0].set(grad[:, 0] / self.f)
        elif self.p==1:  # stereographic
                grad = grad.at[:, 0].set(grad[:, 0] / self.f / (1 + (self.rt_img[:, 0] / 2 * self.f) ** 2))
        elif self.p==2:  # orthographic
                grad = grad.at[:, 0].set(grad[:, 0] / self.f / jnp.sqrt(1 - (self.rt_img[:, 0] * self.f) ** 2))
        elif self.p==3:  # equisolid
                grad = grad.at[:, 0].set(grad[:, 0] / self.f / jnp.sqrt(1 - (self.rt_img[:, 0] / 2 / self.f) ** 2))
        else:  # not defined
                print('Projection "' + self.proj + '" was not defined.')
                exit(-1)
        return grad


    # def compare_gradients(self, Grad):
    #     polar_frames = jnp.dstack(
    #         (jnp.vstack(
    #             (jnp.cos(self.rt_img[:, 1]), jnp.sin(self.rt_img[:, 1]))).T,
    #          jnp.vstack((-jnp.sin(self.rt_img[:, 1]), jnp.cos(self.rt_img[:, 1]))).T))
    #
    #     Grad_lnr = jnp.sum((polar_frames.reshape(-1, 2) * Grad.reshape(-1, 1)).reshape(-1, 2, 2), axis=1)  # coordinate transformation
    #
    #     Grad_lnr[:, 0] = Grad_lnr[:, 0] * self.f
    #
    #     n = int(jnp.sqrt(Grad.shape[0]))
    #     Gradt = Grad_lnr[:, 0].reshape(n, n)
    #     Gradp = Grad_lnr[:, 1].reshape(n, n)
    #     gradt = self.grad_lnr[:, 0].reshape(n, n)
    #     gradp = self.grad_lnr[:, 1].reshape(n, n)
    #
    #     fig1, ax1 = plt.subplots()
    #     pc1 = plt.pcolormesh(Gradt, shading='auto')
    #     ax1.axis('equal')
    #     fig1.colorbar(pc1)
    #     plt.show()
    #
    #     fig2, ax2 = plt.subplots()
    #     pc2 = plt.pcolormesh(Gradp, shading='auto')
    #     ax2.axis('equal')
    #     fig2.colorbar(pc2)
    #     plt.show()
    #
    #     fig1, ax1 = plt.subplots()
    #     pc1 = plt.pcolormesh(gradt, shading='auto')
    #     ax1.axis('equal')
    #     fig1.colorbar(pc1)
    #     plt.show()
    #
    #     fig2, ax2 = plt.subplots()
    #     pc1 = plt.pcolormesh(gradp, shading='auto')
    #     ax2.axis('equal')
    #     fig2.colorbar(pc1)
    #     plt.show()
    #
    #     fig1, ax1 = plt.subplots()
    #     pc1 = plt.pcolormesh(gradt-Gradt, shading='auto')
    #     ax1.axis('equal')
    #     fig1.colorbar(pc1)
    #     plt.show()
    #
    #     fig2, ax2 = plt.subplots()
    #     pc1 = plt.pcolormesh(gradp-Gradp, shading='auto')
    #     ax2.axis('equal')
    #     fig2.colorbar(pc1)
    #     plt.show()
    #
    #     return Grad_lnr