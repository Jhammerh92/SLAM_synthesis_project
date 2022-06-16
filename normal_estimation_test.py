import numpy as np
import open3d as o3d
import copy
import os
import sys
import time

source = o3d.io.read_point_cloud(r'/Users/JHH/Library/Mobile Documents/com~apple~CloudDocs/DTU/10Semester/Syntese/SLAM_synthesis_project/data_test/velodynevlp16/data_pcl/0_frame_218964.770383.pcd')
# estimate normals to better do ICP and 
t1 = time.perf_counter()
source.estimate_normals(
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 5, max_nn = 100))
    # search_param = o3d.geometry.KDTreeSearchParamKNN(knn= 100))
    # search_param = o3d.geometry.KDTreeSearchParamRadius(radius = 1))
# normals are expected to point toward the Lidar scanner location. 
t2 = time.perf_counter()
source.orient_normals_towards_camera_location(np.array([0,0,1.0]))
print(t2 - t1)


# source.orient_normals_to_align_with_direction()
# source.orient_normals_consistent_tangent_plane(50)

o3d.visualization.draw_geometries([source])