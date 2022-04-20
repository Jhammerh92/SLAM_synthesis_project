import numpy as np
import struct
import open3d as o3d
import os
import sys

from load_paths import * # loads paths to data

""" Use this script to convert a KITTI sequence of .bin files to .pcd with estimated normals, to work with in open3d"""

# SEQUENCE NUMBER TO CONVERT
# seq_n = 4

def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    # estimate normals to better do ICP and 
    pcd.estimate_normals(
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 1, max_nn = 100))
    # normals are expected to point toward the Lidar scanner location. 
    pcd.orient_normals_towards_camera_location(np.array([0,0,2.0]))
    return pcd

# seq = "{:02d}".format(seq_n)

# path_to_seq = r"/Volumes/HAMMER LAGER 2 TB/KITTI_LiDAR/dataset/sequences/"
# path_to_bin = path_to_seq + "/" + seq + "/velodyne/"

os.chdir(path_to_bins)
bins_ls = os.listdir()
if not os.path.isdir("pcds"): os.mkdir("pcds") # if pcds is not a dir, create it


N = len(bins_ls) - 1 # - 1 because of the folder pcds we just created
print(N)
print("Starting process")
for b_n in range(N):
    bin_name = "{:06d}.bin".format(b_n)
    pcd_name = "pcds/{:06d}.pcd".format(b_n)
    if os.path.isfile(pcd_name):
        continue
    if b_n % 20 == 0: 
        perc_done = b_n/N
        print("Working on {} of {}. Percent done: {:%}".format(b_n, N, perc_done))
    pcd = convert_kitti_bin_to_pcd(bin_name)
    o3d.io.write_point_cloud(pcd_name, pcd)

print("sequence {} has been converted to .pcd".format(seq))
