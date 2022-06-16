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


vis = o3d.visualization.Visualizer()
vis.create_window()
current = load_pcd_from_index(0, path = PATH_TO_PCDS, is_downsampled=False, voxel_size=VOXEL_SIZE)
vis.add_geometry(current)

vis.get_render_option().load_from_json(path_to_cwd + 'render_options_vis.json')
# load_view_point(path_to_cwd + 'viewpoint.json', vis)


i = 0
# N = 550
while True:
    # time.sleep(0.1)
    print("frame: {}".format(i))

    next_pcd = load_pcd_from_index(i, path = PATH_TO_PCDS, is_downsampled=False, voxel_size=VOXEL_SIZE)
    # source.paint_uniform_color(([0, 0, 0]))
    current.points = next_pcd.points
    current.normals = next_pcd.normals

    vis.update_geometry(current)
    vis.poll_events()
    vis.update_renderer()
    i += 1
    if (i == TOTAL_PCDS) or (i == N): i = 0
vis.run()