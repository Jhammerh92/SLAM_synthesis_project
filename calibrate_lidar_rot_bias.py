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
path = r'/Volumes/HAMMER DATA 2TB/DTU_LIDAR_20220523/20220523_122236/pcds/velodynevlp16/data_pcl'
# path = r'/Users/JHH/Library/Mobile Documents/com~apple~CloudDocs/DTU/10Semester/Syntese/SLAM_synthesis_project/data_drone_flyvning/velodynevlp16/data_pcl'

# os.chdir(path)
pcd_ls = [file for file in os.listdir(path) if file.endswith('.pcd')]
pcd_ls.sort(key=lambda x: "{:06d}".format(int(x.split('_')[0])) )
# pcd_ls = [path + pcd for pcd in pcd_ls]

index = range(450,550)
vox_size = 1.0
calib_frame = load_pcd_from_index(449, path = path, is_downsampled=True, voxel_size=vox_size)
transforms = []
z_rots = []
transform = np.eye(4)
transform[1,3] = -2.0
for i in index:

    next_frame = load_pcd_from_index(i, path = path, is_downsampled=True, voxel_size=vox_size)

    init_trans = transform
    # draw_registration_result(calib_frame, next_frame, init_trans)
    transform = estimate_p2pl(calib_frame, next_frame, init_trans=init_trans, use_coarse_estimate=True, max_iteration=30).transformation
    # draw_registration_result(calib_frame, next_frame, transform)
    z_rot = np.rad2deg(np.arctan2(transform[0,3], - transform[1,3]))
    z_rots.append(z_rot)
    calib_frame = next_frame
    print(z_rot)
z_rots = np.asarray(z_rots)

print(np.mean(z_rots), z_rots )

# should the translation also be included?
# if input("Save calibration?: ").lower() == "y":
#     np.savetxt('lidar_pose_20220523.txt', calib_transform)
#     np.savetxt('lidar_height_20220523.txt', np.array([lidar_height2]))

