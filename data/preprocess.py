import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from manopth.manolayer import ManoLayer
from manopth import demo
import torch.nn as nn
import torch
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def prepare_datalists(root_dir):
    datalist = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if len(filenames) == 0:
            continue
        datalist.append(os.path.relpath(dirpath, root_dir))
    return datalist

def plk2pcd(path,filename,side):
    batch_size = 10
    # Select number of principal components for pose space
    ncomps = 45
    # Initialize MANO layer
    mano_layer = ManoLayer(
    mano_root='./manopth/mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=True, side=side)
    f = open(f'{path}{filename}', 'rb')
    hand_info = pickle.load(f, encoding='latin1')
    f.close()
    theta = nn.Parameter(torch.FloatTensor(hand_info['poseCoeff']).unsqueeze(0))
    beta = nn.Parameter(torch.FloatTensor(hand_info['beta']).unsqueeze(0))
    trans = nn.Parameter(torch.FloatTensor(hand_info['trans']).unsqueeze(0))
    hand_verts, hand_joints = mano_layer(theta, beta)
    kps3d = hand_joints / 1000.0 + trans.unsqueeze(1) # in meters
    hand_transformed_verts = hand_verts / 1000.0 + trans.unsqueeze(1) 
    points=np.concatenate((kps3d[0].detach().numpy(), hand_transformed_verts[0].detach().numpy()), axis=0)
    return points

def save_pkls_as_pcd(rootPath,leftOrRight,path,topath):
    """
    This function load pickles from HOI4d:handpose dataset, convert it to a point cloud and save it
    Parameters
    ----------
    rootPath + leftOrRight + path: location of the pickle files
    topath : location to save the point cloud
    """
    logging.info(f"Converting pickle files to point cloud")
    side=leftOrRight[leftOrRight.rfind('_')+1:]
    pcd = [plk2pcd(os.path.join(rootPath,leftOrRight,path),filename,side) for filename in os.listdir(os.path.join(rootPath,leftOrRight,path))]
    pcd_tuples = [tuple(point) for point in pcd]
    # ZY20210800004/H4/C8/N03/S71/s01/T2
    path=path.replace('\\','/')
    name=path.replace('/','_')
    np.save(f"{topath}{side}_{name}_pcd.npy", np.array(pcd_tuples, dtype=object))

def save_all_pkl_as_pcd(path, topath):
    """
    This function load all pkl files from HOI4D:handpose dataset, convert them to point clouds and save them
    Parameters
    ----------
    path : location of the pkl files
    topath : location to save the point clouds
    """
    logging.info("Starting to process all pkl files")
    
    leftlist=prepare_datalists(os.path.join(path, 'refinehandpose_left'))
    rightlist=prepare_datalists(os.path.join(path, 'refinehandpose_right'))
    for leftpath in leftlist:
        save_pkls_as_pcd(path,'refinehandpose_left',leftpath,topath)      
    for rightpath in rightlist:
        save_pkls_as_pcd(path,'refinehandpose_right',rightpath,topath) 
    logging.info("Finished processing all pkl files")

if __name__ == "__main__":
    raw_data_path = './msr/testDepth/'
    pcd_data_path = './msr/testpcd/'
    save_all_depth_as_pcd(raw_data_path, pcd_data_path)

    # raw_data_path = './hoi4d/handpose/'
    # pcd_data_path = './hoi4d/handposePcd/'
    # save_all_pkl_as_pcd(raw_data_path, pcd_data_path)