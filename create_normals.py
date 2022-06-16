import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys
import pyquaternion
from scipy.spatial.transform import Rotation

from load_paths import *
from functions import *

from thread_pool_loader import _PointCloudTransmissionFormat
import concurrent.futures

from progress_bar import *


"""
to create normals on a set of PCDS that do not have normals
"""

def create_normals_pcd_set(path):
    if path[-1] != '/':
        path += '/'
    pcd_ls = [file for file in os.listdir(path) if file.endswith('.pcd')]
    pcd_ls.sort(key=lambda x: "{:06d}".format(int(x.split('_')[0])) )
    pcd_ls = [path + pcd for pcd in pcd_ls]
    # pcd_name = path + "{:06d}.pcd".format(index)

    start_idx = 0
    print(f"Creating normals on .pcd in {path}")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(estimate_normals_pcd, pcd_path) for pcd_path in pcd_ls[start_idx:]]

        # for i, (future, pcd_name) in enumerate(zip(futures, pcd_ls[start_idx:])):
        for i,f in enumerate( concurrent.futures.as_completed(futures) ):
            progress_bar(i, len(futures))
            # if os.path.isfile(pcd_name):
            #     continue
            # pcd = future.result().create_pointcloud()
            # o3d.io.write_point_cloud(pcd_name, pcd)

    # for i, pcd_name in enumerate( pcd_ls[start_idx:]):
    #     progress_bar(i, len(pcd_ls[start_idx:]))
    #     # if os.path.isfile(pcd_name):
    #     #     continue
    #     pcd = estimate_normals_pcd(pcd_name, mp=False)
    #     # o3d.io.write_point_cloud(pcd_name, pcd)

    print("\ndone")

def estimate_normals_pcd(pcd_path, mp=True) -> _PointCloudTransmissionFormat:
    # pcd = load_pcd_from_index(index, path)
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.normals) > 0:
        print(f"File: {pcd_path}, already has normals and is skipped...")
        return
    pcd.estimate_normals(
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 5, max_nn = 100))
    # normals are expected to point toward the Lidar scanner location. 
    pcd.orient_normals_towards_camera_location(np.array([0.0,0.0,1.0]))
    # o3d.io.write_point_cloud(pcd_path, pcd)
    # if mp:
        # return _PointCloudTransmissionFormat(pcd)
    # else:
    o3d.io.write_point_cloud(pcd_path, pcd)
    return


if __name__ == "__main__":
    # path = r'/Users/JHH/Library/Mobile Documents/com~apple~CloudDocs/DTU/10Semester/Syntese/SLAM_synthesis_project/data_test/velodynevlp16/data_pcl/'
    path =  r'/Volumes/HAMMER DATA 2TB/DTU_LIDAR_20220523/20220523_125726/pcds_2'
    # path =  r'/Volumes/KINGSTON/Drone_LiDAR_VLP16/data_drone_flyvning/velodynevlp16/data_pcl'

    
    create_normals_pcd_set(path)