#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@ Description:
@ Date     : 2024/05/21 11:31:03
@ Author   : sunyifan
@ Version  : 1.0
"""

import numpy as np
from .graphic import getProjectionMatrix


def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def in_frustum(p_orig, viewmatrix):
    # bring point to screen space
    p_view = transformPoint4x3(p_orig, viewmatrix)

    if p_view[2] <= 0.2:
        return None
    return p_view


def transformPoint4x4(p, matrix):
    matrix = np.array(matrix).flatten(order="F")
    x, y, z = p
    transformed = np.array(
        [
            matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
            matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
            matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
            matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15],
        ]
    )
    return transformed


def transformPoint4x3(p, matrix):
    matrix = np.array(matrix).flatten(order="F")
    x, y, z = p
    transformed = np.array(
        [
            matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
            matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
            matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
        ]
    )
    return transformed


# covariance = RS[S^T][R^T]
def computeCov3D(scale, mod, rot):
    # create scaling matrix
    S = np.array(
        [[scale[0] * mod, 0, 0], [0, scale[1] * mod, 0], [0, 0, scale[2] * mod]]
    )

    # normalize quaternion to get valid rotation
    # we use rotation matrix
    R = rot

    # compute 3d world covariance matrix Sigma
    M = np.dot(R, S)
    cov3D = np.dot(M, M.T)

    return cov3D


def computeCov2D(mean, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix):
    # The following models the steps outlined by equations 29
    # and 31 in "EWA Splatting" (Zwicker et al., 2002).
    # Additionally considers aspect / scaling of viewport.
    # Transposes used to account for row-/column-major conventions.
    """ 
    计算协方差矩阵的变换 

    Args:
        mean (np.array): 3DGS椭球在世界坐标系下的中心点 
        focal_x (float): focal length in x x方向的焦距 对应J中的nx 
        focal_y (float): focal length in y y方向的焦距 对应J中的ny 
        tan_fovx (float): tan of field of view in x x方向的视场角 
        tan_fovy (float): tan of field of view in y y方向的视场角 
        cov3D (np.array): 3D covariance matrix 世界坐标系下3DGS椭球的协方差矩阵 
        viewmatrix (np.array): 4x4 view matrix 从世界坐标系到相机坐标系的变换矩阵(观测变换矩阵) 
    """

    # 找到中心点在相机坐标系中的位置 在他的附件才是符合雅克比矩阵的条件的
    t = transformPoint4x3(mean, viewmatrix)

    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = min(limx, max(-limx, txtz)) * t[2]
    t[1] = min(limy, max(-limy, tytz)) * t[2]

    # 计算雅克比矩阵 
    J = np.array(
        [
            [focal_x / t[2], 0, -(focal_x * t[0]) / (t[2] * t[2])],
            [0, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2])],
            [0, 0, 0], # 正交矩阵的最后也就不考虑z方向了 
        ]
    )

    # 按公式从世界坐标系下的协方差矩阵求到视椎被压缩后的状态 
    W = viewmatrix[:3, :3] # (3, 3) 
    T = np.dot(J, W) # JW 

    cov = np.dot(T, cov3D) # JWV_k^''
    cov = np.dot(cov, T.T) # JWV_k^''W^TJ^T (2, 2)

    # Apply low-pass filter
    # Every Gaussia should be at least one pixel wide/high
    # Discard 3rd row and column
    # 现在的cov就是2维的协方差矩阵了 即 cov = 
    # \begin{bmatrix} 
    # \sigma_x^2 & \sigma_xy \\ 
    # \sigma_yx & \sigma_y^2 
    # \end{bmatrix}
    cov[0, 0] += 0.3
    cov[1, 1] += 0.3
    return [cov[0, 0], cov[0, 1], cov[1, 1]]


if __name__ == "__main__":
    p = [2, 0, -2]
    proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
    projmatrix = getProjectionMatrix(**proj_param)
    transformed = transformPoint4x4(p, projmatrix)
    print(transformed)
