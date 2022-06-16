import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys
import pyquaternion
import cv2

import timeit
from numba import jit

from functions import *
from load_paths import * # loads paths to data




# def create_vertex_map(pcd):
#     object_points = np.asarray(pcd.points)

#     # fx = 1 # for 360 (actually 180) degree fov
#     # fy = 90.80712071513639 # for 26.8 degree fov
#     # cx = 0.0
#     # cy = 0.0
#     # cameraMatrix = np.array([[fx, 0,0],[ 0,fy, 0],[cx,cy,1]])
#     # image_points, _ = cv2.projectPoints(object_points, np.eye(3), np.zeros((3,1)), cameraMatrix, 0)

#     depth = np.linalg.norm(object_points, axis = 1)
#     pitch = np.arcsin(object_points[:,2] / depth)
#     yaw = np.arctan2(object_points[:,1], object_points[:,0])

#     FOV = (np.max(pitch)- np.min(pitch)) #np.deg2rad(26.8)
#     FOV_down = np.min(pitch)# np.deg2rad(-24.2)

#     # FOV_down = np.deg2rad(-26.8)
#     # FOV_up = np.deg2rad(2.0)
#     # FOV = FOV_up + abs(FOV_down)

#     h = 64
#     w = 2048
#     # w = 1024

#     u = (w) * (1/2 * (1 - yaw / np.pi))
#     v = (h-1) * (1 - (pitch - FOV_down) / FOV)

#     # vertex_map = np.full((row_scale, col_scale), np.nan)
#     # vertex_map = [[0 for _ in range(w)] for _ in range(h)]
#     # for u,v, d in zip(u.astype(np.int64),v.astype(np.int64), depth):
#     #     vertex_map[ v][ u ] = d

#     vertex_map = np.zeros((h, w))
#     for u,v, d in zip(u.astype(np.int64),v.astype(np.int64), depth):
#         vertex_map[v, u] = d

#     # @jit(cache=True, nopython=True)
#     # def iterate_image_points(U,V,d):
#     #     # vertex_map = np.full((row_scale, col_scale), np.nan)
#     #     vertex_map = np.zeros((h, w))
#     #     for i in range(len(V)):
#     #         vertex_map[V[i], U[i]] = d[i]
#     #     return vertex_map

#     # vertex_map = iterate_image_points(u.astype(np.int64), v.astype(np.int64), depth)
#     return vertex_map

# path = r'/Users/JHH/Library/Mobile Documents/com~apple~CloudDocs/DTU/10Semester/Syntese/SLAM_synthesis_project/data_test/velodynevlp16/data_pcl'
pcd = load_pcd_from_index(0, path=path_to_pcds)

# print(timeit.timeit("create_vertex_map(pcd)","from __main__ import create_vertex_map, pcd", number=1 ))

# draw_pcd(pcd)

# print(image_points)

plt.imshow(create_vertex_map(pcd), cmap="rainbow")
plt.show()
