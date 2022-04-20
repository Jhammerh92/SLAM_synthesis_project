import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys
import pyquaternion

from functions import *
from load_paths import * # loads paths to data
from slam import *



SLAM = PG_SLAM()


# for i in range(500,N):
for i in range(775, N):
    print(i)
    pcd = load_pcd_from_index(i, path_to_pcds, is_downsampled=False, voxel_size=0.5)
    SLAM.update(pcd, downsample_keyframes=True)

print("done")
SLAM.vis.run()