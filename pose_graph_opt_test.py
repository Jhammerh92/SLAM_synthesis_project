import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys
import pyquaternion

from scipy.spatial.transform import Rotation

from functions import *
from load_paths import * # loads paths to data



def optimize(pose_graph, optimize_all = False):
    optimization_options = o3d.pipelines.registration.GlobalOptimizationOption(
                                        max_correspondence_distance=0.5,
                                        edge_prune_threshold=0.25,
                                        preference_loop_closure=2.0,
                                        reference_node=0)
    print("optimizing...")
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                optimization_options)



# create af set of synthetic odometry points in 3D along a 10x10 square.
size = 20
steps = 20
# X = np.linspace(0,size, steps, endpoint=True)
# # X = np.linspace(0,size, steps, endpoint=True)

# XX = np.r_[X, np.full(5, 10), np.flip(X), np.zeros(5)]
# YY = np.r_[np.zeros(5),X, np.full(5, 10), np.flip(X)]
# ZZ = np.zeros_like(XX)

# C = np.c_[XX,YY, ZZ]
# C = np.delete(C,[4,9,14],0)


STD_ROT = 0.1
MU_ROT = 0.00
STD_TRANS = 0.002
MU_TRANS = [0.00,0.00,0.0]

init_pose = np.eye(4)

ODOMETRY = [init_pose]
ODOMETRY_NOISE = [init_pose]
TRANSFORMS = []
TRANSFORMS_NOISE = []
C = []
C_noise = []
C.append(ODOMETRY[-1][:3,3])
C_noise.append(ODOMETRY_NOISE[-1][:3,3])

for i in range(steps):
    transform = np.eye(4)
    transform_noise = np.eye(4)
    if i % (4) == 1 and i != 0:
        angle = np.pi/2
    else:
        angle = 0
    # angle = 0
    transform[:3,:3] = Rotation.from_rotvec(angle * np.array([0, 0, 1])).as_matrix()
    transform_noise[:3,:3] = Rotation.from_rotvec((angle + np.random.normal(MU_ROT ,STD_ROT, 1)) * np.array([0, 0, 1])).as_matrix()
    
    transform[0,3] = size/(steps-1)
    transform_noise[0,3] = size/(steps-1) 
    transform_noise[:3,3] += np.array([np.random.normal(MU_TRANS[0],STD_TRANS), np.random.normal(MU_TRANS[1],STD_TRANS), np.random.normal(MU_TRANS[2],STD_TRANS) ])
    # print(transform)
    ODOMETRY.append(   ODOMETRY[-1] @ transform)
    TRANSFORMS.append(transform)
    ODOMETRY_NOISE.append(ODOMETRY_NOISE[-1] @ transform_noise)
    TRANSFORMS_NOISE.append(transform_noise)

    C.append(ODOMETRY[-1][:3,3])
    C_noise.append(ODOMETRY_NOISE[-1][:3,3])

    # print(C[-1])
    # print(C_noise[-1])
    # print(ODOMETRY[-1])
    # print(ODOMETRY_NOISE[-1])

C = np.asarray(C)
C_noise = np.asarray(C_noise)














POSE_GRAPH = o3d.pipelines.registration.PoseGraph()

# pose_graph_node = o3d.pipelines.registration.PoseGraphNode(ODOMETRY_NOISE[0])
# POSE_GRAPH.nodes.append(pose_graph_node)

for i,odo in enumerate(ODOMETRY_NOISE):
    pose_graph_node = o3d.pipelines.registration.PoseGraphNode(odo)
    POSE_GRAPH.nodes.append(pose_graph_node)
    # optimize(POSE_GRAPH)

    if i == len(ODOMETRY_NOISE)-1: # one less  odometry edge
        break
    edge = o3d.pipelines.registration.PoseGraphEdge(
                                    i+1,
                                    i,
                                    TRANSFORMS_NOISE[i],
                                    uncertain = False)
    POSE_GRAPH.edges.append(edge)



# loop closure edge
edge = o3d.pipelines.registration.PoseGraphEdge(
                                    16,
                                    0,
                                    np.eye(4),
                                    confidence = 1,
                                    uncertain = True)
POSE_GRAPH.edges.append(edge)
edge = o3d.pipelines.registration.PoseGraphEdge(
                                    17,
                                    1,
                                    np.eye(4),
                                    confidence = 1,
                                    uncertain = True)
POSE_GRAPH.edges.append(edge)
edge = o3d.pipelines.registration.PoseGraphEdge(
                                    18,
                                    2,
                                    np.eye(4),
                                    confidence = 1,
                                    uncertain = True)
POSE_GRAPH.edges.append(edge)
edge = o3d.pipelines.registration.PoseGraphEdge(
                                    19,
                                    3,
                                    np.eye(4),
                                    confidence = 1,
                                    uncertain = True)
POSE_GRAPH.edges.append(edge)
edge = o3d.pipelines.registration.PoseGraphEdge(
                                    20,
                                    4,
                                    np.eye(4),
                                    confidence = 1,
                                    uncertain = True)
POSE_GRAPH.edges.append(edge)



pose_graph_C_pre = []
pose_graph_odometry_pre = []
for node in POSE_GRAPH.nodes:
    pose_graph_odometry_pre.append(node.pose)
    pose_graph_C_pre.append(node.pose[:3,3])

pose_graph_C_pre = np.asarray(pose_graph_C_pre)

optimize(POSE_GRAPH)

pose_graph_C = []
pose_graph_odometry = []
for node in POSE_GRAPH.nodes:
    pose_graph_odometry.append(node.pose)
    pose_graph_C.append(node.pose[:3,3])

pose_graph_C = np.asarray(pose_graph_C)





# plot of synthetic data

fig, ax = plt.subplots(1)
ax.plot(C[:,0],C[:,1],'.-',label="Ground")
ax.plot(C_noise[:,0], C_noise[:,1],'.-',label="Added noise")
ax.plot(pose_graph_C_pre[:,0], pose_graph_C_pre[:,1],'.-',label="Pose Graph Before optimization")
ax.plot(pose_graph_C[:,0], pose_graph_C[:,1],'.-',label="Pose Graph After Optimization")
ax.legend()
ax.axis('equal')


plt.show()