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
visual_pcd = o3d.io.read_point_cloud('000000.pcd')
vis.add_geometry(visual_pcd)
load_view_point(path_to_cwd + 'viewpoint.json', vis)

heuristic_trans=np.array([[1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])


# calculate continious transforms
N = 4
egomotion_transforms = []
for i in range(N - 1):
    print("frame: {}".format(i))
    source_name = "{:06d}.pcd".format(i)
    target_name = "{:06d}.pcd".format(i+1)
    # load the pcds
    try:
        source = target
    except:
        source = o3d.io.read_point_cloud(source_name)
    target = o3d.io.read_point_cloud(target_name)

    # update visuals
    visual_pcd.points = target.points
    visual_pcd.normals = target.normals

    vis.update_geometry(visual_pcd)

    reg = estimate_p2pl(source, target, heuristic_trans)
    egomotion_transforms.append(reg.transformation) 
    heuristic_trans = reg.transformation # using the previous found transformation as this is the most likely in a constant motion, either straight or turning

    # load_view_point(path_to_cwd + 'viewpoint.json', vis)
    vis.poll_events()
    vis.update_renderer()
    # draw_registration_result(source, target, reg.transformation)
# vis.run()
# print(egomotion_transforms)


# calculate poses from transforms
# make into point cloud
# pose = o3d.geometry.PointCloud()
# pose.points = o3d.utility.Matrix4dVector(np.eye(4))
# pose = o3d.geometry.TriangleMesh.create_coordinate_frame()
init_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
poses = [init_pose]

for trans in egomotion_transforms:
    # print(np.linalg.inv(trans))
    next_pose = np.dot(poses[-1], trans)
    poses.append(next_pose)

# extract positions and orientations
O = []
p = []
for ps in poses:
    O.append(ps[:-1, :-1])
    p.append(ps[:-1, -1])

# save positions and orientations
T = np.asarray(egomotion_transforms)
p = np.asarray(p)
O = np.asarray(O)

print(T)

# make positions into a point cloud
P = o3d.geometry.PointCloud()
P.points = o3d.utility.Vector3dVector(p)


save_processed_data(**{'positions':p, 'Orientiations':O, 'pos_pcd':P, "Transforms":T})

# create axis that show orientation
axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
axis.transform(poses[-1])



# o3d.visualization.draw_geometries([P, axis])