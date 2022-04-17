import numpy as np
import struct
import open3d as o3d

def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    pcd.estimate_normals(
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 1, max_nn = 100))
    pcd.orient_normals_towards_camera_location(np.array([0,0,2.0]))
    return pcd

path = r"/Volumes/HAMMER LAGER 2 TB/KITTI_LiDAR/dataset/sequences/00/velodyne/000000.bin"
print()
test_pcd = convert_kitti_bin_to_pcd(path)

# Save to whatever format you like
# o3d.io.write_point_cloud("000000.pcd", test_pcd)

o3d.visualization.draw_geometries([test_pcd])