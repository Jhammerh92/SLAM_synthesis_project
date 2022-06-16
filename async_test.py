import timeit
import asyncio

from functions import *
import load_paths

path_to_pcds = load_paths.path_to_pcds
print(path_to_pcds)
t = timeit.Timer('load_pcd_from_index(0, "/Users/JHH/KITTI/sequences/08/velodyne/pcds/", is_downsampled=False, voxel_size=0.05)', 'from functions import load_pcd_from_index')

print(t.repeat(10,100))