"""
Script that handles the undistortion of lidar scan taken while moving
at high relative velocity, compared to the frequency of the scanning.

The theory used for this method is taken from:
"Increased Accuracy For Fast Moving LiDARS: Correction of Distorted Point Clouds"
- Tobias Renzler, Michael Stolz, Markus Schratter

"""



import numpy as np
import open3d as o3d
# import matplotlib.pyplot as plt
import copy
import os
import sys
import pyquaternion
from scipy.spatial.transform import Rotation
import re
import time

from load_paths import *
from functions import *




def undistort_pcd(pcd, transformation, pitch_correction_deg=0.0): # omskriv til at bruge en transform matrice
    """
    Script that handles the undistortion of lidar scan taken while moving
    at high relative velocity, compared to the frequency of the scanning.

    The theory used for this method is taken from:
    "Increased Accuracy For Fast Moving LiDARS: Correction of Distorted Point Clouds"
    - Tobias Renzler, Michael Stolz, Markus Schratter

    with and addition for a pitched lidar, and rotating multiple points fast, using complex vectors.

    Written by Jakob Hammer Hedemann
    """
    undistorted_pcd = copy.deepcopy(pcd)
    delta_time = 1.0# 0.915520-0.714907 # time stamps of the files, GET THEM FROM THE FILES
    # speed = translation/delta_time
    # velocity = transformation[:3,3]
    elevation_correction_angle = np.deg2rad(pitch_correction_deg) #  np.divide( velocity[2] , velocity[1] , out = np.zeros(1), where=abs(velocity[2]) > 0)
    speed = translation_dist_of_transform(transformation, timestep=delta_time)
    omega = angular_rotation_of_transform(transformation, timestep=delta_time, axis='z')

    points = np.asarray(pcd.points)
    points_imag = points * np.array([1.0, 1.0j, 0.0])

    azimuth = np.reshape(np.arctan2(points[:,0], points[:,1]) + (points[:,0] < 0) * 2 * np.pi , (len(points), 1))
    # correction term is a normalized azimuth
    correction_term = 1 - (azimuth/(2*np.pi))
    delta = (correction_term * omega * delta_time)/2
    # adjust distance to make the movement more correrct by calculating it as the arc length
    arc_term = np.divide(abs(np.sin(delta)) , abs(delta),out=np.ones_like(delta), where=abs(delta)>0.0 )
    s = ( -correction_term )* speed * delta_time * arc_term # the negative "velocity vector"
    s_mat = np.zeros_like(points)
    s_mat[:,1] = np.cos(elevation_correction_angle) * s.ravel()
    elevation_correction =  np.sin(elevation_correction_angle) * s.ravel() * correction_term.ravel()

    active_rot_points = rotate_points_3D(s_mat, -delta)
    passive_rot_points = rotate_points_3D(points, 2*delta)
    

    undistorted_points = active_rot_points + passive_rot_points 
    undistorted_points[:,2] += elevation_correction 
    undistorted_pcd.points = o3d.utility.Vector3dVector(undistorted_points)

    return undistorted_pcd

# only rotates about z at the moment
def rotate_points_3D(points, angles, axis='z'):
    # if axis== 'z':
    imag_points = np.reshape(points[:,0] + points[:,1] * 1j ,(-1,1))
    
    imag_angles = np.exp(angles *1j)
    M = imag_angles * imag_points
    rotated_points = np.c_[np.real(M), np.imag(M), points[:,2]]
    return rotated_points


if __name__ == "__main__":
    # path_to_pcds=  r'/Volumes/HAMMER DATA 2TB/DTU_LIDAR_20220523/20220523_124156/pcds/velodynevlp16/data_pcl'
    # calib_lidar_pose = np.loadtxt('lidar_pose_20220523.txt')

    # estimated_vel = 40.0/3.6
    
    idx = 2000
    pcd1 = load_pcd_from_index(idx, PATH_TO_PCDS)#.transform(CALIB_LIDAR_POSE)
    pcd2 = load_pcd_from_index(idx + 1, PATH_TO_PCDS)#.transform(CALIB_LIDAR_POSE)
    reg = estimate_p2pl(pcd1.voxel_down_sample(0.5), pcd2.voxel_down_sample(0.5), use_coarse_estimate=True)
    translation = translation_dist_of_transform(reg.transformation)
    delta_time = 0.20 # 0.915520-0.714907 # time stamps of the files
    speed = translation/delta_time
    omega = angular_rotation_of_transform(reg.transformation, timestep=delta_time)# np.rad2deg(np.arctan2(transform[0,3], - transform[1,3]))
    print(f"Translation: {translation:.2f} m -> speed at 5 Hz scan rate (scan time={delta_time:.2f}s): {speed:.2f} m/s")
    print(f"This corresponds to a distortion of {speed*delta_time:.2f} m between start and end of scan")
    print(f"Omega is: {omega:.2f}")

    print(reg.transformation[:3,3])
    undistorted_pcd = undistort_pcd(pcd1, reg.transformation, pitch_correction_deg=2.0)

    ds = 0.01
    points = copy.deepcopy(pcd1.voxel_down_sample(ds))
    corrected_points = copy.deepcopy(pcd1.voxel_down_sample(ds))
    point = points.select_by_index([0]).paint_uniform_color([1.0,0.0,0.0])

    # to seperate old from new by 1 cm 
    adjust = np.eye(4)
    adjust[2,3] = -0.01

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().load_from_json(path_to_cwd + 'render_options_vis.json')
    vis.add_geometry(pcd1.transform(adjust).voxel_down_sample(ds).paint_uniform_color([0.0,0.0,1.0]))
    vis.add_geometry(undistorted_pcd.transform(adjust).voxel_down_sample(ds).paint_uniform_color([0.0,1.0,0.0]))
    # vis.add_geometry(pcd1)
    # vis.add_geometry(point)
    vis.add_geometry(corrected_points)




    corrected_points_list = []
    for i in range(0, len(points.points)):
        point.points = points.select_by_index(range(0,i+1)).points


        pos = np.asarray(point.points)
        # get the azimuth of the points
        azimuth = np.arctan2(pos[i,0], pos[i,1]) if pos[i,0] >= 0 else np.arctan2(pos[i,0], pos[i,1]) + 2 * np.pi
        #correction term from azimuth
        correction_term = 1 - (azimuth/(2*np.pi))
        # need to make sure delta is correct
        delta = (correction_term * omega * delta_time)/2
        # adjust distance to make the movement more correrct by calculating it as the arc length
        arc_term = abs(np.sin(delta)) / abs(delta)
        s = np.array([[0.0,1.0,0.0]]) *( -correction_term )* speed * delta_time * arc_term
        active_rot_matrix = Rotation.from_rotvec(-delta*np.array([0.0,0.0,1.0])).as_matrix() @ s.T
        passive_rot_matrix = Rotation.from_rotvec(2*delta*np.array([0.0,0.0,1.0])).as_matrix()

        this_point = np.asarray(points.select_by_index([i]).points)
        corrected_point = active_rot_matrix + passive_rot_matrix @ this_point.T
        corrected_points_list.append(corrected_point.T.flatten())
        corrected_points_np = np.asarray(corrected_points_list)
        corrected_points.points = o3d.utility.Vector3dVector(corrected_points_np)

        print(f" Dot #{i},\t\t azimuth {np.rad2deg(azimuth):>5.2f},\t correction term: {correction_term:.2f},\t delta={delta:.5f}", end='\r')


        vis.update_geometry(corrected_points.paint_uniform_color([1.0,0.0,0.0]))
        vis.poll_events()
        vis.update_renderer()
    
    vis.run()