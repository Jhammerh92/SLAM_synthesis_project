import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import os
import sys
import pyquaternion

from functions import *
from load_paths import * # loads paths to data
from slam import *


path = r"/Volumes/HAMMER LAGER 2 TB/KITTI ALL Sequences/sequences/poses/"

for seq in range(11):
    path_txt = path + "{:02}.txt".format(seq)
    pose,_ = get_pose_ground_truth_KITTI(path_txt)
    fig, ax = plt.subplots(1)
    plot_odometry_2D(pose, ax)

plt.show()

