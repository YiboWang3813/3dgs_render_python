#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@ Description:
@ Date     : 2024/05/20 17:20:00
@ Author   : sunyifan
@ Version  : 1.0
"""

import math
import numpy as np
from tqdm import tqdm
from loguru import logger
from math import sqrt, ceil

from render_python import computeColorFromSH
from render_python import computeCov2D, computeCov3D
from render_python import transformPoint4x4, in_frustum
from render_python import getWorld2View2, getProjectionMatrix, ndc2Pix, in_frustum


class Rasterizer:
    def __init__(self) -> None:
        pass

    def forward(
        self,
        P,  # int, num of guassians
        D,  # int, degree of spherical harmonics
        M,  # int, num of sh base function
        background,  # color of background, default black
        width,  # int, width of output image
        height,  # int, height of output image
        means3D,  # ()center position of 3d gaussian
        shs,  # spherical harmonics coefficient
        colors_precomp,
        opacities,  # opacities
        scales,  # scale of 3d gaussians
        scale_modifier,  # default 1
        rotations,  # rotation of 3d gaussians
        cov3d_precomp,
        viewmatrix,  # matrix for view transformation
        projmatrix,  # *(4, 4), matrix for transformation, aka mvp
        cam_pos,  # position of camera
        tan_fovx,  # float, tan value of fovx
        tan_fovy,  # float, tan value of fovy
        prefiltered,
    ) -> None:

        focal_y = height / (2 * tan_fovy)  # focal of y axis
        focal_x = width / (2 * tan_fovx)

        # run preprocessing per-Gaussians
        # transformation, bounding, conversion of SHs to RGB
        logger.info("Starting preprocess per 3d gaussian...")
        preprocessed = self.preprocess(
            P,
            D,
            M,
            means3D,
            scales,
            scale_modifier,
            rotations,
            opacities,
            shs,
            viewmatrix,
            projmatrix,
            cam_pos,
            width,
            height,
            focal_x,
            focal_y,
            tan_fovx,
            tan_fovy,
        )

        # produce [depth] key and corresponding guassian indices
        # sort indices by depth
        depths = preprocessed["depths"]
        point_list = np.argsort(depths)

        # render
        logger.info("Starting render...")
        out_color = self.render(
            point_list,
            width,
            height,
            preprocessed["points_xy_image"],
            preprocessed["rgbs"],
            preprocessed["conic_opacity"],
            background,
        )
        return out_color

    def preprocess(
        self,
        P,
        D,
        M,
        orig_points,
        scales,
        scale_modifier,
        rotations,
        opacities,
        shs,
        viewmatrix,
        projmatrix,
        cam_pos,
        W,
        H,
        focal_x,
        focal_y,
        tan_fovx,
        tan_fovy,
    ):

        rgbs = []  # rgb colors of gaussians
        cov3Ds = []  # covariance of 3d gaussians
        depths = []  # depth of 3d gaussians after view&proj transformation
        radii = []  # radius of 2d gaussians
        conic_opacity = []  # covariance inverse of 2d gaussian and opacity
        points_xy_image = []  # mean of 2d guassians
        for idx in range(P):
            # make sure point in frustum
            p_orig = orig_points[idx]
            p_view = in_frustum(p_orig, viewmatrix)
            if p_view is None:
                continue
            depths.append(p_view[2])

            # transform point, from world to ndc
            # Notice, projmatrix already processed as mvp matrix
            p_hom = transformPoint4x4(p_orig, projmatrix) # 观测变换 + 投影变换 
            p_w = 1 / (p_hom[3] + 0.0000001)
            # p_proj这个点已经在正方体里了 
            p_proj = [p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w] # 归一化 使他齐次坐标中最后一个是1 

            # compute 3d covarance by scaling and rotation parameters
            scale = scales[idx]
            rotation = rotations[idx]
            cov3D = computeCov3D(scale, scale_modifier, rotation)
            cov3Ds.append(cov3D)

            # compute 2D screen-space covariance matrix
            # based on splatting, -> JW Sigma W^T J^T
            cov = computeCov2D(
                p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix
            )

            # invert covarance(EWA splatting)
            det = cov[0] * cov[2] - cov[1] * cov[1]
            if det == 0:
                depths.pop()
                cov3Ds.pop()
                continue
            det_inv = 1 / det
            conic = [cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv]
            conic_opacity.append([conic[0], conic[1], conic[2], opacities[idx]])

            # compute radius, by finding eigenvalues of 2d covariance
            # transfrom point from NDC to Pixel
            mid = 0.5 * (cov[0] + cov[1])
            lambda1 = mid + sqrt(max(0.1, mid * mid - det))
            lambda2 = mid - sqrt(max(0.1, mid * mid - det))
            my_radius = ceil(3 * sqrt(max(lambda1, lambda2)))
            point_image = [ndc2Pix(p_proj[0], W), ndc2Pix(p_proj[1], H)] # 把均值从NDC坐标转换到像素坐标 

            radii.append(my_radius)
            points_xy_image.append(point_image)

            # convert spherical harmonics coefficients to RGB color
            sh = shs[idx]
            result = computeColorFromSH(D, p_orig, cam_pos, sh)
            rgbs.append(result)

        return dict(
            rgbs=rgbs,
            cov3Ds=cov3Ds,
            depths=depths,
            radii=radii,
            conic_opacity=conic_opacity,
            points_xy_image=points_xy_image,
        )

    def render(
        self, point_list, W, H, points_xy_image, features, conic_opacity, bg_color
    ):

        out_color = np.zeros((H, W, 3))
        pbar = tqdm(range(H * W))

        # loop pixel
        for i in range(H):
            for j in range(W):
                pbar.update(1)
                pixf = [i, j]
                C = [0, 0, 0]

                # loop gaussian
                for idx in point_list:

                    # init helper variables, transmirrance
                    T = 1 # 开始肯定没有被遮挡 T就是1 

                    # Resample using conic matrix
                    # (cf. "Surface Splatting" by Zwicker et al., 2001)
                    xy = points_xy_image[idx]  # center of 2d gaussian 当前2d高斯椭球的中心 
                    d = [
                        xy[0] - pixf[0],
                        xy[1] - pixf[1],
                    ]  # distance from center of pixel [dx, dy] 
                    # conic_opacity = [\sigma_x^-1, \sigma_xy^-1, \sigma_y^-1, opacity透明度]
                    con_o = conic_opacity[idx]
                    # power就是 -\frac{1}{2}(x-\mu)T\Sigma^-1(x-\mu)的二维形态
                    power = (
                        -0.5 * (con_o[0] * d[0] * d[0] + con_o[2] * d[1] * d[1])
                        - con_o[1] * d[0] * d[1]
                    )
                    if power > 0: # 像素点离高斯球太远了 没有影响了 
                        continue

                    # Eq. (2) from 3D Gaussian splatting paper.
                    # Compute color
                    # 计算透明度 高斯椭球自身的透明度 con_o[3]
                    # e^{power} 是这个高斯椭球对像素点的影响程度 
                    alpha = min(0.99, con_o[3] * np.exp(power))
                    if alpha < 1 / 255: # 太小就不考虑了 
                        continue

                    # 计算T 就是不被遮挡的概率 
                    test_T = T * (1 - alpha)
                    if test_T < 0.0001: # 如果前面的椭球非常有能量已经占满这个像素了 那后面的椭球也就不用了
                        break

                    # Eq. (3) from 3D Gaussian splatting paper.
                    color = features[idx]
                    for ch in range(3):
                        C[ch] += color[ch] * alpha * T # 按照颜色加权公式把这个高斯椭球的颜色叠加到这个像素上 

                    T = test_T # T没有到极限 把他给下一个高斯椭球 接着遍历 

                # get final color
                for ch in range(3):
                    # alpha blending 如果到了最后一个高斯椭球 然而不透明度还有一定的限度(上限是1) 就用背景颜色把他补上
                    out_color[j, i, ch] = C[ch] + T * bg_color[ch] 

        return out_color


if __name__ == "__main__":
    # set guassian
    pts = np.array([[2, 0, -2], [0, 2, -2], [-2, 0, -2]])
    n = len(pts)
    shs = np.random.random((n, 16, 3))
    opacities = np.ones((n, 1))
    scales = np.ones((n, 3))
    rotations = np.array([np.eye(3)] * n)

    # set camera
    cam_pos = np.array([0, 0, 5])
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
    viewmatrix = getWorld2View2(R=R, t=cam_pos)
    projmatrix = getProjectionMatrix(**proj_param)
    projmatrix = np.dot(projmatrix, viewmatrix)
    tanfovx = math.tan(proj_param["fovX"] * 0.5)
    tanfovy = math.tan(proj_param["fovY"] * 0.5)

    # render
    rasterizer = Rasterizer()
    out_color = rasterizer.forward(
        P=len(pts),
        D=3,
        M=16,
        background=np.array([0, 0, 0]),
        width=700,
        height=700,
        means3D=pts,
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        scale_modifier=1,
        rotations=rotations,
        cov3d_precomp=None,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        cam_pos=cam_pos,
        tan_fovx=tanfovx,
        tan_fovy=tanfovy,
        prefiltered=None,
    )

    import matplotlib.pyplot as plt

    plt.imshow(out_color)
    plt.show()
