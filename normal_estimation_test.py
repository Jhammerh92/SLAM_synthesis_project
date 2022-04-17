import numpy as np
import open3d as o3d
import copy
import os
import sys

source = o3d.io.read_point_cloud('000000.pcd')
# estimate normals to better do ICP and 
source.estimate_normals(
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 1, max_nn = 100))
# normals are expected to point toward the Lidar scanner location. 
source.orient_normals_towards_camera_location(np.array([0,0,2.0]))


# source.orient_normals_to_align_with_direction()
# source.orient_normals_consistent_tangent_plane(50)

o3d.visualization.draw_geometries([source])