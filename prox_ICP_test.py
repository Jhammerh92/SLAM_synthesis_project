import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import copy
import os
import sys
import pyquaternion

from functions import *
from load_paths import * # loads paths to data
from slam import *


os.chdir(path_to_pcds)
# source = o3d.io.read_point_cloud('000007.pcd')
# source = load_pcd_from_index(533)
# target = load_pcd_from_index(1294)
source = load_pcd_from_index(532, is_downsampled=True)
target = load_pcd_from_index(1294, is_downsampled=True)

threshold = 0.5
trans_init = np.array([ [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

rot = Rotation.from_rotvec([0, 0, np.deg2rad(0)])
print(rot.as_matrix())
trans_init[:3,:3] = rot.as_matrix()

# draw_registration_result(source, target, trans_init)

evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold )
print(evaluation)
draw_registration_result(source, target, trans_init)

# needs normals to do point to plane estimations
print("Iterating..")
# reg_p2p = estimate_p2pl(source, target, trans_init, use_coarse_estimate=True, max_iteration=100)
reg_p2p = estimate_p2pl_spin_init(source, target, max_iteration=100)
print("Done!")


print(reg_p2p)
print(reg_p2p.transformation)


draw_registration_result(source, target, reg_p2p.transformation)
# draw_registration_result(source, target, trans_init)

# evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, reg_p2p.transformation)
# print(evaluation)