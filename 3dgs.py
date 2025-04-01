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
        focal_x = width / (2 * tan_fovx)  # focal of x axis 

        # run preprocessing per-Gaussians
        # transformation, bounding, conversion of SHs to RGB
        logger.info("Starting preprocess per 3d gaussian...")
        preprocessed = self.preprocess(P, D, M, means3D, scales, scale_modifier, 
                                       rotations, opacities, shs,
                                       viewmatrix, projmatrix, cam_pos, width, height, 
                                       focal_x, focal_y, tan_fovx, tan_fovy,)

        # produce [depth] key and corresponding guassian indices
        # sort indices by depth
        depths = preprocessed["depths"]
        point_list = np.argsort(depths) # 按高斯椭球的z坐标从小到大排序 得到排序后所有高斯椭球的索引

        # render
        logger.info("Starting render...")
        out_color = self.render(point_list, width, height, preprocessed["points_xy_image"],
                                preprocessed["rgbs"], preprocessed["conic_opacity"], background,)
        
        return out_color

    def preprocess(self, P, D, M, orig_points, scales, scale_modifier, rotations, opacities, shs,
                   viewmatrix, projmatrix, cam_pos, W, H, focal_x, focal_y, tan_fovx, tan_fovy,):
        """ Preprocess 
        
        Args: 
            P (int): 高斯椭球的数量 
            D (int): 球谐函数的阶数 
            M (int): 球谐函数基函数的数量 
            orig_points (np.ndarray): 所有高斯椭球的中心坐标 (世界坐标系) (n, 3) 
            scales (np.ndarray): 所有高斯椭球的放缩因子 (n, 3)
            scale_modifier (float): 放缩因子修改器 默认为1 
            rotations (np.ndarray): 所有高斯椭球的旋转矩阵 默认为眼阵也就是不旋转 (n, 3, 3) 
            opacities (np.ndarray): 所有高斯椭球的透明度 (n, 1)
            shs (np.ndarray): 所有高斯椭球的球谐函数系数矩阵 (n, 16, 3) 
            viewmatrix (np.ndarray): 视角变换矩阵 (4, 4) 
            projmatrix (np.ndarray): 投影变换矩阵与视角变换矩阵的乘积矩阵 (4, 4) 
            cam_pos (np.ndarray): 相机位置 (3,) 
            W (int): 输出图像的宽度 
            H (int): 输出图像的高度 
            focal_x (float): 相机焦距 (x方向) 
            focal_y (float): 相机焦距 (y方向) 
            tan_fovx (float): 相机水平视角的正切值 
            tan_fovy (float): 相机垂直视角的正切值 

        Returns: 
            preprocessed (dict): 包含预处理后的数据 """

        rgbs = []  # rgb colors of gaussians 收集所有高斯椭球的颜色 
        cov3Ds = []  # covariance of 3d gaussians 收集所有高斯椭球在世界坐标系下的协方差矩阵 
        depths = []  # depth of 3d gaussians after view&proj transformation 收集所有高斯椭球在相机坐标系下的z坐标
        radii = []  # radius of 2d gaussians 收集所有高斯椭球在图像坐标系下的半径
        conic_opacity = []  # covariance inverse of 2d gaussian and opacity 收集所有高斯椭球在图像坐标系下的协方差矩阵的逆和透明度
        points_xy_image = []  # mean of 2d guassians 收集所有高斯椭球在图像坐标系下的均值

        for idx in range(P):
            # make sure point in frustum
            p_orig = orig_points[idx] # (3,) 
            p_view = in_frustum(p_orig, viewmatrix) # (3,) 当前高斯椭球在相机坐标系中的坐标
            if p_view is None:
                continue
            depths.append(p_view[2]) # depths收集每个高斯椭球在相机坐标系中的z坐标

            # transform point, from world to ndc
            # Notice, projmatrix already processed as mvp matrix
            p_hom = transformPoint4x4(p_orig, projmatrix) # (4,) 当前高斯椭球在ndc坐标系中的中心坐标(未归一化)
            p_w = 1 / (p_hom[3] + 0.0000001)
            p_proj = [p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w] # 归一化 (3,) 当前高斯椭球在ndc坐标系中的中心坐标(归一化)

            # compute 3d covarance by scaling and rotation parameters
            scale = scales[idx] # (3,) 当前高斯椭球的缩放因子 默认是(1, 1, 1)  
            rotation = rotations[idx] # (3, 3) 当前高斯椭球的旋转矩阵 默认是眼阵  
            cov3D = computeCov3D(scale, scale_modifier, rotation) # (3, 3) 当前高斯椭球在世界坐标系下的协方差矩阵
            cov3Ds.append(cov3D) # cov3Ds收集每个高斯椭球在世界坐标系下的协方差矩阵 

            # compute 2D screen-space covariance matrix
            # based on splatting, -> JW Sigma W^T J^T
            cov = computeCov2D(
                p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix
            ) # 得到2D屏幕空间的协方差矩阵 (3,) 

            # invert covarance(EWA splatting)
            det = cov[0] * cov[2] - cov[1] * cov[1]
            if det == 0: # 协方差矩阵行列式为0 则跳过 
                depths.pop()
                cov3Ds.pop()
                continue
            det_inv = 1 / det
            conic = [cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv] # 得到2D平面协方差矩阵的逆 
            conic_opacity.append([conic[0], conic[1], conic[2], opacities[idx]]) # 组合这个逆以及该高斯椭球的透明度 

            # compute radius, by finding eigenvalues of 2d covariance
            mid = 0.5 * (cov[0] + cov[1])
            lambda1 = mid + sqrt(max(0.1, mid * mid - det))
            lambda2 = mid - sqrt(max(0.1, mid * mid - det))
            my_radius = ceil(3 * sqrt(max(lambda1, lambda2)))
            radii.append(my_radius)

            # transfrom point from NDC to Pixel 把高斯椭球的均值从NDC空间转化到像素空间 
            point_image = [ndc2Pix(p_proj[0], W), ndc2Pix(p_proj[1], H)] 
            points_xy_image.append(point_image)

            # convert spherical harmonics coefficients to RGB color
            sh = shs[idx] # (16, 3) 
            result = computeColorFromSH(D, p_orig, cam_pos, sh) # (3,) 高斯椭球的颜色 
            rgbs.append(result)

        return dict(
            rgbs=rgbs, # [(3,) * n] 
            cov3Ds=cov3Ds, # [(3, 3) * n] 
            depths=depths, # [float * n] 
            radii=radii, # [float * n] 
            conic_opacity=conic_opacity, # [[float * 4] * n] 
            points_xy_image=points_xy_image, # [[float * 2] * n]
        ) 

    def render(self, point_list, W, H, points_xy_image, features, conic_opacity, bg_color):
        """ 
        Render, and output an image 
        
        Args:
            point_list (list): 按高斯椭球在相机坐标系中z坐标从小到大排序后的索引列表 
            W (int): 输出图像的宽度 
            H (int): 输出图像的高度 
            points_xy_image (list): 所有高斯椭球在图像坐标系下的均值 [[float * 2] * n]
            features (list): 所有高斯椭球的颜色 [(3,) * n] 
            conic_opacity (list): 所有高斯椭球在图像坐标系下的协方差矩阵的逆以及透明度 [[float * 4] * n] 
            bg_color (np.ndarray): 背景颜色 默认为黑色 (3,)
        """

        out_color = np.zeros((H, W, 3)) # 输出图像初始化为空白画布 
        pbar = tqdm(range(H * W))

        # loop pixel
        for i in range(H):
            for j in range(W): # 遍历每一个像素 
                pbar.update(1)
                pixf = [i, j]
                C = [0, 0, 0]

                # loop gaussian
                for idx in point_list: # 遍历每一个高斯椭球 

                    # init helper variables, transmirrance 光可以传到s点的概率 
                    T = 1 # 开始位置s=0 光线肯定没有被阻碍 则在s之前光线没有被阻碍的概率为1 

                    # Resample using conic matrix
                    # (cf. "Surface Splatting" by Zwicker et al., 2001)
                    xy = points_xy_image[idx]  # center of 2d gaussian 当前高斯椭球的均值 

                    # distance from center of pixel [dx, dy] 从当前高斯椭球中心到当前像素的距离
                    # TODO 这个顺序好像反了 points_xy_image用的坐标轴的xy顺序和ij的顺序反了 
                    d = [xy[0] - pixf[0], xy[1] - pixf[1],]  
                    
                    # 结合当前高斯椭球的协方差矩阵的逆以及前面的距离计算power 
                    con_o = conic_opacity[idx]
                    # power就是 -\frac{1}{2}(x-\mu)T\Sigma^-1(x-\mu)的二维形态
                    power = (
                        -0.5 * (con_o[0] * d[0] * d[0] + con_o[2] * d[1] * d[1])
                        - con_o[1] * d[0] * d[1]
                    )
                    if power > 0: # 像素点离高斯球太远了 没有影响了 power太大就不要 
                        continue

                    # Eq. (2) from 3D Gaussian splatting paper.
                    # Compute color
                    alpha = min(0.99, con_o[3] * np.exp(power)) # opacity -> alpha 当前高斯椭球的透明度 
                    if alpha < 1 / 255: # 透明度太小说明这个高斯椭球对该像素的影响太小 直接不要他  
                        continue

                    # 计算这个高斯椭球的T 并据此判断是否停止遍历后序的高斯椭球 
                    test_T = T * (1 - alpha) # T每过一个高斯椭球都会变小一点 
                    if test_T < 0.0001: # 如果前面的椭球非常有能量已经占满这个像素了 那后面的椭球也就不用了
                        break

                    # Eq. (3) from 3D Gaussian splatting paper.
                    color = features[idx] # 拿到这个高斯椭球的颜色 
                    for ch in range(3):
                        C[ch] += color[ch] * alpha * T # 按照颜色加权公式把这个高斯椭球的颜色叠加到这个像素上 

                    T = test_T # T没有到极限 把他给下一个高斯椭球 接着遍历 

                # get final color
                for ch in range(3):
                    # alpha blending 如果到了最后一个高斯椭球 然后T还有一定的额度 就用背景颜色来补上 
                    out_color[j, i, ch] = C[ch] + T * bg_color[ch] 

        return out_color


if __name__ == "__main__":
    # set guassian
    pts = np.array([[2, 0, -2], [0, 2, -2], [-2, 0, -2]]) # (3, 3) 3个高斯椭球的中心点 
    n = len(pts)
    shs = np.random.random((n, 16, 3)) # (3, 16, 3) 3个高斯椭球每个椭球有16个球谐函数系数 
    opacities = np.ones((n, 1)) # (3, 1) 3个高斯椭球的不透明度 
    scales = np.ones((n, 3)) # (3, 3) 3个高斯椭球的放缩因子
    rotations = np.array([np.eye(3)] * n) # (3, 3, 3) 3个高斯椭球的旋转 

    # set camera
    cam_pos = np.array([0, 0, 5]) # (3,) 相机位置 
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]) # (3, 3) 相机旋转矩阵 
    proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
    viewmatrix = getWorld2View2(R=R, t=cam_pos) # (4, 4) 观测矩阵 (相机的外参矩阵) (世界坐标系到相机坐标系转换矩阵)
    projmatrix = getProjectionMatrix(**proj_param) # (4, 4) 投影矩阵 
    projmatrix = np.dot(projmatrix, viewmatrix) # 综合投影矩阵和观测矩阵 
    tanfovx = math.tan(proj_param["fovX"] * 0.5)
    tanfovy = math.tan(proj_param["fovY"] * 0.5)

    # render
    rasterizer = Rasterizer()
    out_color = rasterizer.forward(
        P=len(pts), # 3 
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
