import logging
import os
import numpy as np
import matplotlib.pyplot as plt
# from .. import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

raw_data_path = './raw/Depth/'
pcd_data_path = './raw/point_clouds/'

def loadDepthMap(path):
    """
    This function reads a depth image from MSR Action3D dataset
    Input:
        path - location of the bin file
    Output:
        depthMap - depth image
    """
    with open(path, 'rb') as fid:
        numFrames, dims = readHeader(fid)
        fileData = np.fromfile(fid, dtype=np.uint32)

    depth = fileData.astype(np.float64)
    depthCountPerMap = np.prod(dims)

    depthMap = []
    for _ in range(numFrames):
        currentDepthData = depth[:depthCountPerMap]
        depth = depth[depthCountPerMap:]
        depthMap.append(currentDepthData.reshape(dims[::-1])[::-1,::-1])

    return depthMap

def readHeader(fid):
    numFrames = np.frombuffer(fid.read(4), dtype=np.uint32)[0]
    dims = np.frombuffer(fid.read(8), dtype=np.uint32)
    return numFrames, dims

def showDepthMap(depthMap):
    """
    This function shows a depth image from MSR Action3D dataset
    Input:
        depthMap - depth image
    """
    depthMap = np.array(depthMap)
    plt.imshow(depthMap)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()

def depth2pcd(depthMap):
    """
    This function convert a depth image from MSR Action3D dataset to a point cloud
    Input:
        depthMap - depth image
    Output:
        pcd - point cloud
    """
    x, y = depthMap.shape
    xx, yy = np.meshgrid(np.arange(x), np.arange(y), indexing='ij')

    mask = depthMap != 0
    xx = xx[mask]
    yy = yy[mask]
    zz = depthMap[mask]
    
    return np.stack((xx, yy, zz), axis=-1)

def save_depth_as_pcd(name):
    """
    This function load a depth image from MSR Action3D dataset, convert it to a point cloud and save it
    Input:
        name - name of the file
    """
    logging.info(f"Converting depth map to point cloud for file {name}")
    depthmap = loadDepthMap(f"{raw_data_path}{name}_sdepth.bin")

    pcd = [depth2pcd(depthmap[i]) for i in range(len(depthmap))]
    pcd_tuples = [tuple(point) for point in pcd]
    
    np.save(f"{pcd_data_path}{name}_pcd.npy", np.array(pcd_tuples, dtype=object))

def save_all_depth_as_pcd():
    """
    This function load all depth images from MSR Action3D dataset, convert them to point clouds and save them
    """
    logging.info("Starting to process all depth images")
    
    for filename in os.listdir(raw_data_path):
        save_depth_as_pcd(filename[:filename.rfind('_')])
    
    logging.info("Finished processing all depth images")

def load_pcd(name):
    """
    This function load a point cloud from MSR Action3D dataset
    Input:
        name - name of the file
    Output:
        pcd - point cloud
    """
    pcd_tuples = np.load(f"{pcd_data_path}{name}_pcd.npy", allow_pickle=True)
    return [np.array(point) for point in pcd_tuples]