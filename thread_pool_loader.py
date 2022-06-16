import numpy as np
import open3d as o3d
# import matplotlib.pyplot as plt
import copy
import os
import sys
# import pyquaternion
import time

from functions import *
from load_paths import * # loads paths to data
from slam import *

# import multiprocessing
import concurrent.futures

class _PointCloudTransmissionFormat:
    def __init__(self, pointcloud: o3d.geometry.PointCloud()):
        self.points = np.array(pointcloud.points)
        self.colors = np.array(pointcloud.colors)
        self.normals = np.array(pointcloud.normals)

    def create_pointcloud(self) -> o3d.geometry.PointCloud():
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(self.points)
        pointcloud.colors = o3d.utility.Vector3dVector(self.colors)
        pointcloud.normals = o3d.utility.Vector3dVector(self.normals)
        return pointcloud

_pool_ = concurrent.futures.ThreadPoolExecutor()

class PCDloader:
    
    def __init__(self, path, start_idx = 0, end_idx = None, voxel_size=0.5):
        self.start_idx = start_idx
        self.cycle = start_idx
        if path[-1] != '/':
            path += '/'
        self.path = path
        self.voxel_size = voxel_size

        self.pcd_ls = [file for file in os.listdir(self.path) if file.endswith('.pcd')]
        # sort by 6 cifre zero padded index i.e. 000001, split by _ or .
        self.pcd_ls.sort(key=lambda x: "{:06d}".format(int(x.replace('.','_').split('_')[0])) )
        if end_idx is None:
            self.end_idx = len(self.pcd_ls)
        else:
            self.end_idx = end_idx

        self.load_next()


    def iterative_loader(self):
        for _ in range(self.start_idx, self.end_idx):
            # yield load_pcd_from_index(self.cycle, is_downsampled=False, voxel_size=0.5)
            # print(f"Frame {self.cycle}") 
            
            # t1_load = time.perf_counter()
            current, current_ds, ts = self.retrieve_next()

            self.cycle += 1
            # t2_load = time.perf_counter()
            try:
                self.load_next()
            except:
                print("no more frames...")  

            # print(f"load time:\t {t2_load-t1_load:2f}s")
            yield current, current_ds, ts
        _pool_.shutdown()

        

    def load_next(self):
        
        pcd_name = self.path + self.pcd_ls[self.cycle]
        args = pcd_name, self.cycle, self.voxel_size
        # print(*args)
        self.next = _pool_.submit(load_pcd_from_name, *args)

    def retrieve_next(self):
        result1,results2, timestamp = self.next.result()
        return result1.create_pointcloud(), results2.create_pointcloud(), timestamp




def load_pcd_from_name(pcd_name, cycle, voxel_size=0.5) -> _PointCloudTransmissionFormat:
    _pcd = o3d.io.read_point_cloud(pcd_name)
    # filtering of the points that are "below" the ground surface i.e. wrong readings, for kitti that is below -1.73
    # x, y are for points on the car itself

    # create vertex map?


    # preprocessing, should be done in separate function
    # x_threshold = 2.0
    # y_threshold = 2.0
    # z_threshold = -200.0
    # r_inner_threshold = 2.0
    # r_outer_threshold = 200.0
    # points = np.asarray(_pcd.points)
    # mask_inner = np.linalg.norm(points[:,:2], axis=1) > r_inner_threshold
    # mask_outer = np.linalg.norm(points[:,:2], axis=1) < r_outer_threshold
    # mask_z = points[:,2] > z_threshold
    # mask = np.logical_and(mask_inner & mask_outer, mask_z)
    # # np.logical_and()
    # _pcd = _pcd.select_by_index(np.where(mask)[0])




    # points = np.asarray(_pcd.points)
    # _pcd = _pcd.select_by_index(np.where(abs(points[:,0]) > x_threshold)[0])
    # points = np.asarray(_pcd.points)
    # _pcd = _pcd.select_by_index(np.where()[0])
    timestamp = None
    if len(pcd_name_split := pcd_name.split('/')[-1].split('_')) > 1 :
        timestamp = float(pcd_name_split[-1].replace('.pcd',''))
    else:
        txt_file = pcd_name.replace(pcd_name.split('/')[-1], '').replace('pcds/', '') + 'timestamps.txt'
        timestamp = get_timestamp_from_txt(cycle, txt_file)



    _pcd_ds = _pcd.voxel_down_sample(voxel_size=voxel_size)
    # _pcd_ds = _pcd.random_down_sample(0.005)
    # _pcd_ds = _pcd.uniform_down_sample(100)
    return _PointCloudTransmissionFormat(_pcd), _PointCloudTransmissionFormat(_pcd_ds), timestamp


def get_timestamp_from_txt(idx, txt):
    with open(txt, 'r') as f: 
        lines = f.readlines()
        timestamp = float(lines[idx])
    return timestamp