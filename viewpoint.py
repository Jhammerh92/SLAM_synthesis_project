import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys

from functions import *
from load_paths import * # loads paths to data


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(filename, vis=None):
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    ctr.convert_from_pinhole_camera_parameters(param)
    # vis.run()
    # vis.destroy_window()
    


if __name__ == "__main__":
    # pcd_data = o3d.data.PCDPointCloud()
    # pcd = o3d.io.read_point_cloud(pcd_data.path)
    pcd = o3d.io.read_point_cloud(path_to_pcds + '000000.pcd')
    save_view_point(pcd, "viewpoint.json")
    print("new view point is saved!")
    # load_view_point(pcd, "viewpoint.json")