import logging
import os
import numpy as np
import matplotlib.pyplot as plt
# from .. import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def disturb_data(clip):
    # 随机生成旋转矩阵
    theta = np.random.uniform(-np.pi/4, np.pi/4)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])

    # 随机生成平移矩阵
    trans_vec = np.random.uniform(-0.1, 0.1, size=(1, 3))

    # 随机生成缩放矩阵
    scale_vec = np.random.uniform(0.9, 1.1, size=(1, 3))
    scale_mat = np.diag(scale_vec[0])

    # 将旋转、平移和缩放矩阵组合成一个变换矩阵
    trans_mat = np.dot(rot_mat, scale_mat)
    # trans_mat[:, 2] = trans_vec

    # 将变换矩阵应用到点云数组上
    return np.dot(clip, trans_mat.T)+trans_vec*clip

def loadDepthMap(path):
    """
    This function reads a depth image from MSR Action3D dataset
    Parameters
    ----------
    path : location of the bin file
    Returns
    -------
    depthMap : depth image
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

def showpcd(pcd):
    """
    This function shows a point cloud from MSR Action3D dataset
    Parameters
    ----------
    pcd : point cloud
    """
    pcd = np.array(pcd)
    plt.scatter(pcd[:, 1], pcd[:, 0], c=pcd[:, 2])
    # plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.colorbar()
    plt.show()

def showDepthMap(depthMap):
    """
    This function shows a depth image from MSR Action3D dataset
    Parameters
    ----------
    depthMap : depth image
    """
    depthMap = np.array(depthMap)
    plt.imshow(depthMap)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()

def depth2pcd(depthMap):
    """
    This function convert a depth image from MSR Action3D dataset to a point cloud
    Parameters
    ----------
    depthMap : depth image
    Returns
    -------
    pcd : point cloud
    """
    x, y = depthMap.shape
    xx, yy = np.meshgrid(np.arange(x), np.arange(y), indexing='ij')

    mask = depthMap != 0
    xx = xx[mask]
    yy = yy[mask]
    zz = depthMap[mask]
    
    return np.stack((xx, yy, zz), axis=-1)

def save_depth_as_pcd(path, topath, name):
    """
    This function load a depth image from MSR Action3D dataset, convert it to a point cloud and save it
    Parameters
    ----------
    path : location of the bin file
    topath : location to save the point cloud
    name : name of the file
    """
    logging.info(f"Converting depth map to point cloud for file {name}")
    depthmap = loadDepthMap(f"{path}{name}_sdepth.bin")

    pcd = [depth2pcd(depthmap[i]) for i in range(len(depthmap))]
    pcd_tuples = [tuple(point) for point in pcd]
    
    np.save(f"{topath}{name}_pcd.npy", np.array(pcd_tuples, dtype=object))

def save_all_depth_as_pcd(path, topath):
    """
    This function load all depth images from MSR Action3D dataset, convert them to point clouds and save them
    Parameters
    ----------
    path : location of the bin files
    topath : location to save the point clouds
    """
    logging.info("Starting to process all depth images")
    
    for filename in os.listdir(path):
        save_depth_as_pcd(path, topath, filename[:filename.rfind('_')])
    
    logging.info("Finished processing all depth images")

def load_pcd(path, name):
    """
    This function load a point cloud from MSR Action3D dataset
    Parameters
    ----------
    path : location of the point cloud
    name : name of the file
    Returns
    -------
    pcd : point cloud
    """
    pcd_tuples = np.load(f"{path}{name}_pcd.npy", allow_pickle=True)
    return [np.array(point) for point in pcd_tuples]

if __name__ == "__main__":
    raw_data_path = './raw/testDepth/'
    pcd_data_path = './raw/testpcd/'
    save_all_depth_as_pcd(raw_data_path, pcd_data_path)