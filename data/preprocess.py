import numpy as np
import matplotlib.pyplot as plt

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
        depthMap.append(currentDepthData.reshape(dims[::-1]))

    return depthMap

def readHeader(fid):
    numFrames = np.frombuffer(fid.read(4), dtype=np.uint32)[0]
    dims = np.frombuffer(fid.read(8), dtype=np.uint32)
    return numFrames, dims

def showDepthMap(depthMap):
    # depthMap - depth map file (matrix of depths)
    depthMap = np.array(depthMap)
    plt.imshow(depthMap)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()