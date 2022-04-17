import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys

from functions import *
from load_paths import * # loads paths to data




os.chdir(path_to_pcds)
vis = o3d.visualization.Visualizer()
vis.create_window()
current = o3d.io.read_point_cloud('000000.pcd')
vis.add_geometry(current)

vis.get_render_option().load_from_json(path_to_cwd + 'render_options_vis.json')
load_view_point(path_to_cwd + 'viewpoint.json', vis)

# o3d.visualization.RenderOption.background_color = np.array([0.0,0.0,0.0])


# N = 4540
for i in range(N):
    print("frame: {}".format(i))
    # next_name = "{:06d}.pcd".format(i)

    # # vis.remove_geometry(source)

    # next_pcd = o3d.io.read_point_cloud(next_name)
    next_pcd = load_pcd_from_index(i, is_downsampled=True, voxel_size=0.1)

    # source.paint_uniform_color(([0, 0, 0]))
    current.points = next_pcd.points
    current.normals = next_pcd.normals
    # vis.add_geometry(source)
    vis.update_geometry(current)
    vis.poll_events()
    vis.update_renderer()
vis.run()