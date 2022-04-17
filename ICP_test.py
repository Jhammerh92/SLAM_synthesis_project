import numpy as np
import open3d as o3d
import copy
import os
import sys

from functions import *
from load_paths import * # loads paths to data



os.chdir(path_to_pcds)
# source = o3d.io.read_point_cloud('000007.pcd')
source = load_pcd_from_index(7)
target = o3d.io.read_point_cloud('000008.pcd')

threshold = 0.2
trans_init = np.array([ [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

# draw_registration_result(source, target, trans_init)

evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
print(evaluation)

# needs normals to do point to plane estimations
print("Iterating..")
reg_p2p = estimate_p2pl(source, target, use_coarse_estimate=False)
print("Done!")


print(reg_p2p)
print(reg_p2p.transformation)

draw_registration_result(source, target, reg_p2p.transformation)

# evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, reg_p2p.transformation)
# print(evaluation)

