# STPCAR
Spatio-Temporal Point Cloud Action Recognition

```
STPCAR/
├── data/
│   ├── raw/
│   │   ├── Depth/
│   │   └── point_clouds/
│   ├── train/
│   ├── test/
│   ├── __init__.py
│   ├── dataset.py
│   └── preprocess.py
├── models/
│   ├── __init__.py
│   ├── Transformer.py
│   ├── P4Convolution.py
│   └── P4Transformer.py
├── utils/
│   ├── __init__.py
│   └── utils.py
├── config.py
├── train.py
├── test.py
└── README.md
```

## Data
使用 MSR-Action3D 数据集，记录了人体动作序列，共包含20个动作类型，10个被试者，每个被试者执行每个动作2或3次。 总共有567个深度图序列。 分辨率为640x240。 用类似于Kinect装置的深度传感器记录数据。

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

## Installation

```
cd utils
python setup.py install
```