import os
import sys

from functions import *
from load_paths import *


path =  r'/Volumes/HAMMER DATA 2TB/DTU_LIDAR_20220523/20220523_125726/omdøb'
pcd_ls = [file for file in os.listdir(path) if file.endswith('.pcd')]
pcd_ls.sort(key=lambda x: "{:06d}".format(int(x.replace('.','_').split('_')[0])) )

os.chdir(path)
new_range = [ i+5192 for i in range(0, len(pcd_ls))]
print(new_range)
# print(pcd_ls)

for name, new_idx in zip(pcd_ls, new_range):
    new_name = f'{new_idx:06d}{name[6:]}'
    print(name, new_name)
    os.rename(name, new_name)
print('renamed')