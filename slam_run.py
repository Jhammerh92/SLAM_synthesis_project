# import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt
import copy
import os
import sys
# import pyquaternion
import time
# import pandas as pd

from settings import *
from functions import *
from load_paths import * # loads paths to data
from slam import *

# import plotting

import thread_pool_loader as tpl


# import faulthandler; faulthandler.enable()

# import multiprocessing
# import concurrent.futures

def main():
    
    # z_rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0.0,0.0,1.0]) * np.deg2rad(-90))
    # z_rot_bias = np.eye(4)
    # z_rot_bias[:3,:3] = z_rot_mat
    # calib_lidar_pose = calib_lidar_pose @ z_rot_bias

    bias_correction={ # this is not really used.
        "x":     0.0,
        "y":     0.0,
        "z":     0.0,
        "pitch": 0.0,
        "roll":  0.0,
        "yaw":   0.0,
    }

    # ground_truth_odometry,_ = plotting.get_pose_ground_truth_KITTI(path_to_pose_txt)
    print(SETTINGS)

    SLAM = PG_SLAM(bias_correction,**slam_params)# CALIB_LIDAR_POSE, undistort_pcd=UNDISTORT_PCD, cam_follow=CAM_FOLLOW, use_loop_closure=USE_LOOP_CLOSURE)
    # SLAM = PG_SLAM(bias_correction, use_loop_closure=True)

    # frames intervals on the 2nd sequence, dtu
    # 250, 950 : 900, 2200 : 2100, end
    PCDLoader = tpl.PCDloader(PATH_TO_PCDS, START_INDEX, END_INDEX, voxel_size=VOXEL_SIZE)

    # SLAM.debug_plane()

    t_start = time.perf_counter()
    for i, pcds in enumerate(PCDLoader.iterative_loader()): # returns both full and downsampled pcd in 'pcds'
        # if i == 1557-1300:
        #     print('FAULT if these been a loop closure added before this point')
        SLAM.update(*pcds, gps_position=None, use_downsampled=True, downsample_keyframes=True)

    SLAM.finalize() #finalize function .. push last keyframe, optimize,update vis .. etc.
    t_end = time.perf_counter()



    print(f"\nSLAM done in {t_end - t_start}s")

    # ax = SLAM.plot_pose_graph()
    # SLAM.plot_odometry(ax)
    # SLAM.plot_non_opt_odometry(ax)
    # ground_truth,_ = get_pose_ground_truth_KITTI(path_to_pose)
    # plot_odometry_2D(ground_truth, ax, arg='KITTI', label='Ground truth')
    
    # ax.legend()
    # plt.show()

    # SLAM.vis.run()
    # local_map = SLAM.generate_local_map(5)
    # draw_pcd(local_map)

    slam_map = SLAM.generate_map()


    to_be_saved = {"SLAM_map": slam_map,
                    "odometry": SLAM.get_odometry(),
                    "non_opt_odometry":SLAM.get_odometry_non_opt(),
                    "poses": SLAM.get_poses(),
                    "keyframe_poses": SLAM.get_keyframe_poses(),
                    'timestamps': SLAM.get_timestamps()
                  }

    save_processed_data(SLAM,**to_be_saved)


    # draw_pcd(slam_map)


if __name__ == "__main__":
    main()
   
   
   
   
