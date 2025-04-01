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
    # 用viewmatrix把点从世界坐标系转到相机坐标系 
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
    """ 用缩放和旋转矩阵计算在世界坐标系下一个高斯椭球的协方差矩阵 """
    # create scaling matrix
    S = np.array(
        [[scale[0] * mod, 0, 0], [0, scale[1] * mod, 0], [0, 0, scale[2] * mod]]
    ) # (3, 3) 

    # normalize quaternion to get valid rotation
    # we use rotation matrix
    R = rot # (3, 3) 

    # compute 3d world covariance matrix Sigma
    M = np.dot(R, S)
    cov3D = np.dot(M, M.T)

    return cov3D # (3, 3) 


def computeCov2D(mean, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix):
    # The following models the steps outlined by equations 29
    # and 31 in "EWA Splatting" (Zwicker et al., 2002).
    # Additionally considers aspect / scaling of viewport.
    # Transposes used to account for row-/column-major conventions.
    """ 
    将高斯椭球的协方差矩阵从世界坐标系变换到2D平面坐标系

    Args:
        mean (np.array): 3D高斯椭球在世界坐标系下的中心点 
        focal_x (float): focal length in x x方向的焦距 对应J中的nx 
        focal_y (float): focal length in y y方向的焦距 对应J中的ny 
        tan_fovx (float): tan of field of view in x x方向的视场角的tan值
        tan_fovy (float): tan of field of view in y y方向的视场角的tan值 
        cov3D (np.array): 3D covariance matrix 3D高斯椭球在世界坐标系下的协方差矩阵 
        viewmatrix (np.array): 4x4 view matrix 从世界坐标系到相机坐标系的变换矩阵(观测变换矩阵) 
    """

    # 找到中心点在相机坐标系中的位置 在他的附近才是符合雅克比矩阵的条件的
    t = transformPoint4x3(mean, viewmatrix) # (3,) 高斯椭球在相机坐标系下的中心点坐标 

    # 约束t的坐标范围 
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
            [0, 0, 0], # 因为要的是2D平面的协方差矩阵 所以不需要z轴分量 
        ]
    )

    # 得到高斯椭球在长方体坐标系下的协方差矩阵 
    W = viewmatrix[:3, :3] # (3, 3) 
    T = np.dot(J, W) # JW 
    cov = np.dot(T, cov3D) # JWV_k^''
    cov = np.dot(cov, T.T) # JWV_k^''W^TJ^T (3, 3)

    # Apply low-pass filter
    # Every Gaussia should be at least one pixel wide/high
    # Discard 3rd row and column
    cov[0, 0] += 0.3
    cov[1, 1] += 0.3

    # 长方体坐标系直接把z不要 只要xy的部分 就能得到2D平面的协方差矩阵 cov2d = cov[:2, :2] 
    # cov2d[0, 1] = cov2d[1, 0] 所以只要返回3个参数即可 
    return [cov[0, 0], cov[0, 1], cov[1, 1]] # \sigma_x^2, \sigma_{xy}, \sigma_y^2 


if __name__ == "__main__":
    p = [2, 0, -2]
    proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
    projmatrix = getProjectionMatrix(**proj_param)
    transformed = transformPoint4x4(p, projmatrix)
    print(transformed)
