import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys

from functions import *
from load_paths import * # loads paths to data

# def load_pcd_from_index(index, path=""):
#     pcd_name = "path + {:06d}.pcd".format(index)
    # return o3d.io.read_point_cloud(pcd_name)


NN=100


def build_pose_graph(folder_with_pcds, index=None):
    odometry_pose = np.eye(4)
    transformation = odometry_pose
    pose_graph = o3d.pipelines.registration.PoseGraph()

    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry_pose))

    positions = []

    potential_closure_idx = set(())

    n_pcds = NN
    for source_idx in range(n_pcds):
        print("source frame: {}".format(source_idx))
        target_idx = source_idx + 1
        # for target_idx in range(source_idx +1, n_pcds, 10):
        #     print("target frame: {}".format(target_idx))
            # load the pcds
            # source_name = "{:06d}.pcd".format(source_idx)
            # target_name = "{:06d}.pcd".format(target_idx)
            # try:
            #     source = target # attpempts to hand target to source instead of loading the same data twice
            # except:
        source = load_pcd_from_index(source_idx, folder_with_pcds, is_downsampled=True)
        target = load_pcd_from_index(target_idx, folder_with_pcds, is_downsampled=True)

        transformation = estimate_p2pl(source, target, transformation).transformation # use previous transformation as initial
        transformation_inv = np.linalg.inv(transformation)
        # if target_idx == source_idx + 1:
        # print(transformation)

        uncertain_flag = False
        # odometry_pose = np.dot(odometry_pose,transformation)
        odometry_pose = odometry_pose @ transformation
        positions.append(odometry_pose[:-1,3])
        # print(odometry_pose)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry_pose))) # the cumulated transformation to given node
        
        # loop closure detection
        distances = calculate_internal_distances(np.asarray(positions))
        proximity_idx = np.where(distances[-1,:] < 10**2)[0]
        proximity_idx_mask = proximity_idx < (source_idx - 100)
        proximity_idx = proximity_idx[proximity_idx_mask]
        # print(positions[-1])
        if any(proximity_idx_mask):
            print("proximity detected")
            for prox_idx in proximity_idx:
                if prox_idx in potential_closure_idx:
                    continue
                # find index of loop closure target and determine transformation, and add edge with uncertain=True
                target_prox = load_pcd_from_index(prox_idx, folder_with_pcds, is_downsampled=True)
                transformation_prox = np.linalg.inv(estimate_p2pl(source, target_prox).transformation)
                uncertain_flag = True
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_idx,
                                                                                prox_idx,
                                                                                transformation_prox, 
                                                                                uncertain=uncertain_flag))
                potential_closure_idx.add(prox_idx)


        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_idx,
                                                                        target_idx,
                                                                        transformation_inv, 
                                                                        uncertain=uncertain_flag))

    return pose_graph


pose_graph = build_pose_graph(path_to_pcds)







print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=0.5,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)






print("Transform points and display")

pcd_combined = o3d.geometry.PointCloud()

for point_id in range(0,NN,10):
    print(point_id)
    
    pcd = load_pcd_from_index(point_id, path_to_pcds)

    pcd.transform(pose_graph.nodes[point_id].pose)
    pcd_combined = pcd_combined + pcd
    if point_id % 25 == 0:
        pcd_combined = pcd_combined.voxel_down_sample(0.2)
        
pcd_combined = pcd_combined.voxel_down_sample(0.2)

o3d.visualization.draw_geometries([pcd_combined])