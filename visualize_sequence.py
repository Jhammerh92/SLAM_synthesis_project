import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys
import time

from settings import *
from functions import *
from load_paths import * # loads paths to data

import thread_pool_loader as tpl


vis = o3d.visualization.Visualizer()
vis.create_window()
current = load_pcd_from_index(0, path = PATH_TO_PCDS, is_downsampled=False, voxel_size=VOXEL_SIZE)
vis.add_geometry(current)

vis.get_render_option().load_from_json(path_to_cwd + 'render_options_vis.json')
# load_view_point(path_to_cwd + 'viewpoint.json', vis)

lidar_pose = CALIB_LIDAR_POSE

init = 850
N = 1500
i = init

loader = tpl.PCDloader(PATH_TO_PCDS, init, N, repeat=True)

while True:
    # time.sleep(0.1)
    print("frame: {}".format(i))
    next_pcd = next(loader.iterative_loader())[0]
    # next_pcd = load_pcd_from_index(i, path = PATH_TO_PCDS, is_downsampled=False, voxel_size=VOXEL_SIZE)
    next_pcd.transform(lidar_pose)
    # IDEA: CREATE VERTEX MAP AND 2D HIST MAP AND SHOW ALSO
    # source.paint_uniform_color(([0, 0, 0]))
    current.points = next_pcd.points
    current.normals = next_pcd.normals

    vis.update_geometry(current)
    if not vis.poll_events():
        break
    vis.update_renderer()
    i += 1
    if (i == TOTAL_PCDS) or (i == N): i = init
vis.run()