import open3d as o3d

from functions import *
from load_paths import *


pcd = load_pcd_from_index(2732, PATH_TO_PCDS)
draw_pcd(pcd)