import numpy as np
import struct
import open3d as o3d
import os
from progress_bar import *

from load_paths import * # loads paths to data
from thread_pool_loader import _PointCloudTransmissionFormat
import concurrent.futures

""" Use this script to convert a KITTI sequence of .bin files to .pcd with estimated normals, to work with in open3d
    call in terminal: 'python3 convert_bins_to_pcd.py -i <inputdir> -o <outputdir>' OR 'python3 convert_bins_to_pcd.py -i <inputdir>'

"""

# SEQUENCE NUMBER TO CONVERT
# seq_n = 4

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
    # estimate normals to better do ICP and 
    pcd.estimate_normals(
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 1, max_nn = 100))
    # normals are expected to point toward the Lidar scanner location. 
    pcd.orient_normals_towards_camera_location(np.array([0,0,2.0]))
    return pcd


def convert_kitti_bin_to_pcd_ThreadPool(binFilePath) -> _PointCloudTransmissionFormat:
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
    # estimate normals to better do ICP and 
    pcd.estimate_normals(
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 1, max_nn = 100))
    # normals are expected to point toward the Lidar scanner location. 
    pcd.orient_normals_towards_camera_location(np.array([0.0,0.0,0.0]))

    return _PointCloudTransmissionFormat(pcd)









# seq = "{:02d}".format(seq_n)

# path_to_seq = r"/Volumes/HAMMER LAGER 2 TB/KITTI_LiDAR/dataset/sequences/"
# path_to_bin = path_to_seq + "/" + seq + "/velodyne/"
def main(argv):


    try: opts, args = getopt.getopt(argv,'hci:o:', ["help"]) # this also handles uknown arguments!
    except:
        print("usage: convert_bins_to_pcd.py -i <inputdir> -o <outputdir> or convert_bins_to_pcd.py -i <inputdir>")
        sys.exit(2)

    create_out_dir = False
    for opt, arg, in opts:
        # print(opt,arg)
        if opt in ("-h", "--help"):
            print("convert_bins_to_pcd.py -i <inputdir> -o <outputdir> or convert_bins_to_pcd.py -i <inputdir>")
            sys.exit(2)
        elif opt in ("-i"):
            path_to_bins = r'{}'.format(arg)
            path_to_pcds =  r'{}'.format(arg) + r'/pcds' # if no out is given, then out is the same as in
        elif opt in ("-o"):
            path_to_pcds = r'{}'.format(arg) + r'/pcds'
        elif opt in ("-c"):
            create_out_dir = True
        # else:
        #     print("option {} is not known".format(opt))
        #     sys.exit(2)


    print("Converting .bin files to .pcd for use with Open3d \n")
            

    print("inputdir is {}".format(path_to_bins))
    print("outputdir is {}".format(path_to_pcds))

    if not os.path.isdir(path_to_bins):
        raise Exception("inputdir {} does not exist".format(path_to_bins))

    if path_to_bins[-1] != "/":
        path_to_bins += "/"


    
    try:
        os.chdir(path_to_pcds) # path is changed to output path 
    except:
        if create_out_dir:
            print("creating outputdir {}".format(path_to_pcds))
            os.makedirs(path_to_pcds)
            try:
                os.chdir(path_to_pcds) # path is changed to output path
            except:
                raise Exception("outputdir was not created")
        else:
            raise Exception("outputdir does not exist")
    




    # if not os.path.isdir("pcds"): os.mkdir("pcds") # if pcds is not a dir, create it
    # os.chdir(path_to_bins)
    bins_ls = [file for file in os.listdir(path_to_bins) if file.endswith('.bin')]# list all the bin files
    N = len(bins_ls) 
    if N == 0:
        raise Exception("No .bin files in inputdir")

    # Might need to sort the order??


    bin_paths = [path_to_bins + bin_name for bin_name in bins_ls]
    pcd_names = [f"{path_to_pcds}/{i:06d}.pcd" for i in range( N )]

    # remove already processed files from the list of files to process
    bin_paths = [bin_path for (bin_path, pcd_name) in zip(bin_paths, pcd_names) if not os.path.isfile(pcd_name) ]
    pcd_names = [pcd_name for pcd_name in pcd_names if not os.path.isfile(pcd_name) ]

    # print(pcd_names[0])
    # print(bin_paths[0])

    # print(f"Files to convert {N}\n")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        print("doing conversion stuff..")
        futures = [executor.submit(convert_kitti_bin_to_pcd_ThreadPool, bin_path) for bin_path in bin_paths]

        for i, (future, pcd_name) in enumerate(zip(futures, pcd_names)):
            progress_bar(i, len(futures))
            # if os.path.isfile(pcd_name):
            #     continue
            pcd = future.result().create_pointcloud()
            o3d.io.write_point_cloud(pcd_name, pcd)


    # print("Starting process")
    # progress_bar(0, N, clear=True)
    # for b_n in range(N):
    #     bin_name = "{:06d}.bin".format(b_n)
    #     pcd_name = "pcds/{:06d}.pcd".format(b_n)
    #     if os.path.isfile(pcd_name):
    #         continue
    #     try:
    #         pcd = convert_kitti_bin_to_pcd(path_to_bins + bin_name)
    #         o3d.io.write_point_cloud(pcd_name, pcd)
    #     except:
    #         print("{} does not exist".format(bin_name))



    print("\nbins have been converted to .pcd in folder {}".format(path_to_pcds))




if __name__ == "__main__":
    main(sys.argv[1:])