"""
Fisheye class defines transformations of coordinates and normals
Torch Implementation
Author: Yuxiang Ma
Date:   04/19/2022
"""
# import numpy as np
import torch

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
        x, y = torch.meshgrid(x_img, y_img)
        self.xy_img = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], 1)  # (x, y) of all points and iterate y first
        self.proj = proj
        self.f = f
        self.p = self.projs.index(proj)

        self.cart2polar()
        self.polar2sph()

    def cart2polar(self):
        # ! On image plane, converting cartesian coordinates to polar coordinates.
        r_img = torch.sqrt(self.xy_img[:, 0] ** 2 + self.xy_img[:, 1] ** 2)
        t_img = torch.arctan2(self.xy_img[:, 1], self.xy_img[:, 0])
        self.rt_img = torch.cat([r_img.view(-1, 1), t_img.view(-1, 1)], 1)

    def polar2cart(self):
        # ! On image plane, converting polar coordinates to cartesian coordinates.
        x_img = self.rt_img[:, 0] * torch.cos(self.rt_img[:, 1])
        y_img = self.rt_img[:, 0] * torch.sin(self.rt_img[:, 1])
        self.xy_img = torch.cat([x_img.view(-1, 1), y_img.view(-1, 1)], 1)

    def polar2sph(self):
        # ! Converting polar coordinates to spherical coordinates (without r)
        self.tp = torch.clone(self.rt_img)
        if self.p==0:       # equidistant
                self.tp[:, 0] = self.rt_img[:, 0] / self.f
        elif self.p==1:     # stereographic
                self.tp[:, 0] = torch.arctan(self.rt_img[:, 0] / 2 / self.f)
        elif self.p==2:     # orthographic
                self.tp[:, 0] = torch.arcsin(self.rt_img[:, 0] / self.f)
        elif self.p==3:     # equisolid
                self.tp[:, 0] = 2 * torch.arcsin(self.rt_img[:, 0] / 2 / self.f)
        else:               # not defined
                print('Projection "' + self.proj + '" was not defined.')
                exit(-1)

    def sph2polar(self):
        # ! Converting polar coordinates to spherical coordinates (without r)
        self.rt_img = torch.clone(self.tp)
        if self.p==0:       # equidistant
                self.rt_img[:, 0] = self.f * self.tp[:, 0]
        elif self.p==1:     # stereographic
                self.rt_img[:, 0] = 2 * self.f * torch.tan(self.tp[:, 0] / 2)
        elif self.p==2:     # orthographic
                self.rt_img[:, 0] = self.f * torch.sin(self.tp[:, 0])
        elif self.p==3:     # equisolid
                self.rt_img[:, 0] = 2 * self.f * torch.sin(self.tp[:, 0] / 2)
        else:               # not defined
                print('Projection "' + self.proj + '" was not defined.')
                exit(-1)

    # these two functions are projections between cartesian coordinate and spherical coordinate in real world
    def cart2sph(self, xyz):
        # cartesian coordinate(xyz) to spherical coordinate(rtp)
        x2y2 = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        r = torch.sqrt(x2y2 + xyz[:, 2] ** 2)
        t = torch.arctan2(torch.sqrt(x2y2), xyz[:, 2])
        p = torch.arctan2(xyz[:, 1], xyz[:, 0])
        return torch.cat([r.view(-1, 1), t.view(-1, 1), p.view(-1, 1)], 1)

    def sph2cart(self, rtp):
        # spherical coordinate(rtp) to cartesian coordinate(xyz)
        z = rtp[:, 0] * torch.cos(rtp[:, 1])
        r = rtp[:, 0] * torch.sin(rtp[:, 1])
        y = r * torch.sin(rtp[:, 2])
        x = r * torch.cos(rtp[:, 2])
        return torch.cat([x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)], 1)

    def convert_norms(self, normals):
        # ! converting norms in real space to gradients(norms) in computational space (x_img, y_img, r)
        # 1, transform normals to spherical coordinates
        # 2, turn normals in spherical coordinates into gradient of ln(r) with regard to (rho, phi) on image plane
        # 3, transform gradient of ln(r) from polar to cartesian (on image plane)

        # transform normals to spherical coordinates
        # local spherical frameworks (sphframes shape 3*3*n)
        sph_frames = torch.stack(
            [torch.cat(
                [torch.sin(self.tp[:, 0]) * torch.cos(self.tp[:, 1]), torch.sin(self.tp[:, 0]) * torch.sin(self.tp[:, 1]),
                 torch.cos(self.tp[:, 0])], 1),
             torch.cat(
                 [torch.cos(self.tp[:, 0]) * torch.cos(self.tp[:, 1]), torch.cos(self.tp[:, 0]) * torch.sin(self.tp[:, 1]),
                  -torch.sin(self.tp[:, 0])], 1),
             torch.cat([-torch.sin(self.tp[:, 1]), torch.cos(self.tp[:, 1]), torch.zeros(self.tp.shape[0])], 1)])
        norm_sph = torch.sum((sph_frames.view(-1, 3) * normals.view(-1, 1)).view(-1, 3, 3), axis=1)

        # turn norms into gradient with respect to (rho, phi) on image plane
        grad_lnr = - norm_sph[:, 1:] / norm_sph[:, 0].view(-1, 1)    # normalization
        grad_lnr[:, 1] = grad_lnr[:, 1] * torch.cos(self.tp[:, 0])           # derivatives with regard to (theta, phi)
        self.grad_lnr  = grad_lnr

        grad_lnr = self.gradient2image(grad_lnr)                        # transform to image plane

        # On image plane, transform gradient of ln(r) from polar to cartesian
        polar_frames = torch.stack(
            [torch.cat([torch.cos(self.rt_img[:, 1]), -torch.sin(self.rt_img[:, 1])], 1).T,
             torch.cat([torch.sin(self.rt_img[:, 1]), torch.cos(self.rt_img[:, 1])], 1).T])

        grad = torch.sum((polar_frames.view(-1, 2)*grad_lnr.view(-1, 1)).view(-1, 2, 2), axis=1)  # coordinate transformation

        return grad

    def gradient2image(self, grad):
        if self.p==0:  # equidistant
                grad[:, 0] = grad[:, 0] / self.f
        elif self.p==1:  # stereographic
                grad[:, 0] = grad[:, 0] / self.f / (1 + (self.rt_img[:, 0] / 2 * self.f) ** 2)
        elif self.p==2:  # orthographic
                grad[:, 0] = grad[:, 0] / self.f / torch.sqrt(1 - (self.rt_img[:, 0] * self.f) ** 2)
        elif self.p==3:  # equisolid
                grad[:, 0] = grad[:, 0] / self.f / torch.sqrt(1 - (self.rt_img[:, 0] / 2 / self.f) ** 2)
        else:  # not defined
                print('Projection "' + self.proj + '" was not defined.')
                exit(-1)
        return grad


    # def compare_gradients(self, Grad):
    #     polar_frames = torch.dstack(
    #         (torch.vstack(
    #             (torch.cos(self.rt_img[:, 1]), torch.sin(self.rt_img[:, 1]))).T,
    #          torch.vstack((-torch.sin(self.rt_img[:, 1]), torch.cos(self.rt_img[:, 1]))).T))
    #
    #     Grad_lnr = torch.sum((polar_frames.view(-1, 2) * Grad.view(-1, 1)).view(-1, 2, 2), axis=1)  # coordinate transformation
    #
    #     Grad_lnr[:, 0] = Grad_lnr[:, 0] * self.f
    #
    #     n = int(torch.sqrt(Grad.shape[0]))
    #     Gradt = Grad_lnr[:, 0].view(n, n)
    #     Gradp = Grad_lnr[:, 1].view(n, n)
    #     gradt = self.grad_lnr[:, 0].view(n, n)
    #     gradp = self.grad_lnr[:, 1].view(n, n)
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