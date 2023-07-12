import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from manopth.manolayer import ManoLayer
from manopth import demo
import torch.nn as nn
import torch
import pickle
# from .. import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_datalists(root_dir):
    datalist = []
    for ZY in os.listdir(root_dir):
        p1 = os.path.join(root_dir, ZY)
        if not os.path.isdir(p1):
            continue
        for H in os.listdir(p1):
            p2 = os.path.join(p1, H)
            if not os.path.isdir(p2):
                continue
            for C in os.listdir(p2):
                p3 = os.path.join(p2, C)
                if not os.path.isdir(p3):
                    continue
                for N in os.listdir(p3):
                    p4 = os.path.join(p3, N)
                    if not os.path.isdir(p4):
                        continue
                    for S in os.listdir(p4):
                        p5 = os.path.join(p4, S)
                        if not os.path.isdir(p5):
                            continue
                        for s in os.listdir(p5):
                            p6 = os.path.join(p5, s)
                            if not os.path.isdir(p6):
                                continue
                            for T in os.listdir(p6):
                                datalist.append(os.path.join(ZY, H, C, N, S, s, T))
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

def load_pcd(path, name):
    """
    This function load a point cloud from MSR Action3D dataset
    Parameters
    ----------
    path : location of the point cloud
    name : name of the fil
    -------
    pcd : point cloud
    """
    pcd_tuples = np.load(f"{path}{name}_pcd.npy", allow_pickle=True)
    return [np.array(point) for point in pcd_tuples]

if __name__ == "__main__":
    raw_data_path = './handpose/'
    pcd_data_path = './handposePcd/'
    save_all_pkl_as_pcd(raw_data_path, pcd_data_path)e
    Returns