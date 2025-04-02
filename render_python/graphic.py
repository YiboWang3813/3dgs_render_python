#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@ Description:
@ Date     : 2024/05/21 11:31:03
@ Author   : sunyifan
@ Version  : 1.0
"""

import math
import numpy as np


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """ 
    获得相机的外参矩阵 (世界坐标系转相机坐标系) (viewmatrix) (观测变换)
    
    Args: 
        R (np.ndarray): 旋转矩阵 (3, 3) 
        t (np.ndarray): 平移向量 (3, 1) 
        translate (np.ndarray): 相机中心的平移量 (3,) 
        scale (float): 相机中心的缩放量 
    
    Returns: 
        Rt (np.ndarray): 外参矩阵 (4, 4) 
    """
    # 组装相机的外参矩阵 
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # 外参矩阵转位姿矩阵 + 平移缩放相机中心 + 位姿矩阵转回外参矩阵 
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)

    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """ 获得投影矩阵 """
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


if __name__ == "__main__":
    p = [2, 0, -2]
    proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
    projmatrix = getProjectionMatrix(**proj_param)
    print(projmatrix)
