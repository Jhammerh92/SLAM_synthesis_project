import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys
import time

from functions import *
from load_paths import * # loads paths to data


# path = r'/Users/JHH/Library/Mobile Documents/com~apple~CloudDocs/DTU/10Semester/Syntese/SLAM_synthesis_project/data_test/velodynevlp16/data_pcl'
# path = r'/Volumes/HAMMER LAGER 2 TB/DTU_LIDAR_20220523/20220523_121111/pcds/velodynevlp16/data_pcl'
path = r'/Volumes/HAMMER DATA 2TB/DTU_LIDAR_20220523/20220523_121111/pcds/'
# path = r'/Users/JHH/Library/Mobile Documents/com~apple~CloudDocs/DTU/10Semester/Syntese/SLAM_synthesis_project/data_drone_flyvning/velodynevlp16/data_pcl'

# os.chdir(path)
pcd_ls = [file for file in os.listdir(path) if file.endswith('.pcd')]
pcd_ls.sort(key=lambda x: "{:06d}".format(int(x.split('_')[0])) )
# pcd_ls = [path + pcd for pcd in pcd_ls]
N = len(pcd_ls)

z_rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0.0,0.0,1.0]) * np.deg2rad(0))
# z_rot_bias = np.eye(4)
# z_rot_bias[:3,:3] = z_rot_mat

calib_frame = load_pcd_from_index(100, path = path, is_downsampled=False, voxel_size=0.1).rotate(z_rot_mat)

calib_frame_crop = crop_pcd(calib_frame,z=(-4,-1))


plane_model, inliers = calib_frame_crop.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
extracted_plane = calib_frame_crop.select_by_index(inliers)
print((plane_model))

# create perfect plane in x, y, z=0 in eg. 50x50 meters
area = (-40.0, 40.0)
X = np.linspace(*area, 501)
Y = X
XX, YY = np.meshgrid(X,Y)
XX = XX.ravel()
YY = YY.ravel()
# ZZ = np.ones_like(XX)* -2.0
ZZ = np.zeros_like(XX)
plane_np = np.c_[XX, YY, ZZ]

fin_np = np.c_[ZZ, XX, YY]

calib_plane = o3d.geometry.PointCloud()
calib_fin = o3d.geometry.PointCloud()
calib_plane.points = o3d.utility.Vector3dVector( plane_np)
calib_fin.points = o3d.utility.Vector3dVector( fin_np)

calib_plane.estimate_normals(
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 5, max_nn = 100))
# normals are expected to point toward the Lidar scanner location. 
calib_plane.orient_normals_towards_camera_location(np.array([0.0,0.0,1.0]))

calib_fin.estimate_normals(
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 5, max_nn = 100))
# normals are expected to point toward the Lidar scanner location. 
# calib_plane.orient_normals_towards_camera_location(np.array([0.0,0.0,1.0]))


init_trans = np.eye(4)
init_trans[2, 3] = plane_model[3] # adjust the plane height of the init transform
extracted_plane.transform(init_trans)
reg_cal = estimate_p2pl(extracted_plane, calib_plane, init_trans=np.eye(4))

calib_transform = copy.deepcopy(reg_cal.transformation)

theta_z = np.arctan2(calib_transform[1,0], calib_transform[0,0])
print(f"rotation about the z-axis {np.rad2deg(theta_z)}")

transformed_calib_frame = copy.deepcopy(calib_frame) 
transformed_calib_frame.transform(calib_transform) 
transformed_calib_frame += calib_fin

transformed_extracted_plane = copy.deepcopy(extracted_plane).transform(calib_transform) 


vis = o3d.visualization.Visualizer()
vis.create_window()
# vis.add_geometry(calib_frame)
# vis.add_geometry((calib_frame + calib_fin.transform(np.linalg.inv(calib_transform))).paint_uniform_color([1.0, 0.0, 0.0]))
vis.add_geometry(transformed_extracted_plane.paint_uniform_color([0.0, 1.0, 0.0]))
vis.add_geometry(calib_plane.paint_uniform_color([0.0, 0.0, 1.0]))
# vis.add_geometry(calib_frame.paint_uniform_color([0.0, 0.0, 1.0]))
vis.add_geometry(extracted_plane.paint_uniform_color([1.0, 0.0, 0.0]))
vis.run()

lidar_height = calib_transform[2, 3]
lidar_height2 = plane_model[3]
print(lidar_height, lidar_height2)
lidar_pose = calib_transform
# print(calib_transform)
print(lidar_pose)


# should the translation also be included?
if input("Save calibration?:  ").lower() == "y":
    np.savetxt('lidar_pose_20220523_alt90.txt', lidar_pose)
    np.savetxt('lidar_height_20220523_alt90.txt', np.array([lidar_height2]))
    print("Saved!")

