import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys

from functions import *
from load_paths import * # loads paths to data


# THIS I NOT FEASIBLE/ PRACTICAL
os.chdir(path_to_pcds)

pcds = []
N = 4540
for i in range(N - 1):
    print("frame: {}".format(i))
    source_name = "{:06d}.pcd".format(i)

    pcds.append(o3d.io.read_point_cloud(source_name))

o3d.visualization.draw_geometries([pcds[200]])
