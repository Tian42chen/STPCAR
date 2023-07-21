# STPCAR
Spatio-Temporal Point Cloud Action Recognition

```
STPCAR/
├── data/
│   ├── msr/
│   │   ├── Depth/
│   │   └── pcd/
│   ├── hoi4d/
│   │   ├── handpose/
│   │   └── handposePcd/
│   ├── __init__.py
│   ├── dataset.py
│   └── preprocess.py
├── models/
│   ├── __init__.py
│   ├── Transformer.py
│   ├── P4Convolution.py
│   └── P4Transformer.py
├── utils/
│   ├── _ext_src/
│   ├── __init__.py
│   ├── pointnet2_utils.py
│   ├── pytorch_utils.py
│   ├── setup.py
│   └── utils.py
├── config.py
├── train.py
├── test.py
└── README.md
```

## Dataset
### MSR Action3D
使用 [MSR Action3D][msr] 数据集，记录了人体动作序列，共包含20个动作类型，10个被试者，每个被试者执行每个动作2或3次。 总共有567个深度图序列。 分辨率为640x240。 用类似于Kinect装置的深度传感器记录数据。

文件名 a01_s01_e01: actions 1 performed by subjects 1 with instances 1

actions 对应的动作类型如下：
1. high arm wave
1. horizontal arm wave
1. hammer
1. hand catch
1. forward punch
1. high throw
1. draw x
1. draw tick
1. draw circle
1. hand clap
1. two hand wave
1. side-boxing
1. bend
1. forward kick
1. side kick
1. jogging
1. tennis swing
1. tennis serve
1. golf swing
1. pick up & throw

### HOI4D
使用[HOI4D](https://www.hoi4d.top/#downLoad)中的Hand Pose数据集。HOI4D 由 2.4M RGB-D 以自我为中心的视频帧组成，超过 4000 个序列，由 9 名参与者与来自 610 个不同室内房间的 16 个类别的 800 个不同对象实例进行交互。Hand Pose数据集根据每个视频中的MANO参数呈现手部姿势。在每个文件中：`.pickle`
- `"poseCoeff"`：指 3 个全局旋转 + 45 个 mano 姿势参数
- `"beta"`：指 10 个马诺形状参数。每个人类ID H*的形状是相同的。
- `"trans"` ：指手在相机框架中的平移
- `"kps2D"`：指每张图像上渲染的手部姿势的21个关键点投影协调。

./ZY2021080000*/H*/C*/N*/S*/s*/T*/ 目录：
- ZY2021080000* refers to the camera ID.
- H* refers to human ID.
- C* refers to object class.
```
mapping = [
    '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
    'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
    'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
]
```
- N* refers to object instance ID.
- S* refers to the room ID.
- s* refers to the room layout ID.
- T* refers to the task ID.

具体内容参考 [HOI4D-Instruction](https://github.com/leolyliu/HOI4D-Instructions) 的说明
## Installation 
### Set up the Python environment
#### MSR-Action3D
```
pip install -r requirements.txt

cd utils
python setup.py install
```
安装依赖，并编译安装 `pointnet2._ext`

<!-- ~~这里使用 github copilot chat 把 cuda 的代码转化为了 c++ 的代码以支持 cpu，但是没有测试过，可能会有问题~~

太多函数是使用 cuda 实现的了, 难以支持 cpu 版本 -->
#### HOI4D
从 [这里](https://github.com/hassony2/manopth) 下载manopth，然后将manopth文件夹放入`data`目录下

### Set up datasets
#### MSR-Action3D
下载MSR-Action3D 深度数据，解压到`data/raw/Depth`目录下，
```
python preprocess.py
```
运行`data/preprocess.py`，将深度图转换为点云数据，保存在`data/raw/point_clouds`目录下。读取深度图借鉴了 [MSR Action3D][msr] 中 matlab 代码

#### HOI4D
下载HOI4D的human hand pose数据，解压到`data/handpose`目录下，
```
python processHandpose.py
```
运行`data/processHandpose.py`，将手部的特征数据通过处理转换为点云数据，保存在`data/handposePcd`目录下。处理手部数据参考了 [manopth](https://github.com/hassony2/manopth) 和 [hoi4d](https://github.com/leolyliu/HOI4D-Instructions) 中的示例代码

## Train
```
python train.py
```
在config.py中设置参数，训练模型  
训练 log 会保存在 `log_path/log.txt` 之中

## Test
```
python test.py
```
在config.py中设置参数，测试模型


 [msr]:https://sites.google.com/view/wanqingli/data-sets/msr-action3d
