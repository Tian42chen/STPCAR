# Hyperparameters
learning_rate = 0.01
momentum = 0.9
weight_decay = 1e-4
batch_size = 16
num_epochs = 30
print_interval=10
# num_epochs = 50
# print_interval = 100
workers=4

lr_milestones=[12, 18]
# lr_milestones=[20, 30]
lr_gamma=0.1
lr_warmup_epochs=10

# Data paths
data_path = './data/raw/testpcd/'

# Model paths
save_model_path = './checkpoints/'

# Model parameters
## P4DConv: spatial
radius=0.7
nsamples=32
spatial_stride=32
## P4DConv: temporal
temporal_kernel_size=3
temporal_stride=2
## embedding: relu
emb_relu=False
# Transformer
dim=1024
depth=5
heads=8
dim_head=128
## output
mlp_dim=2048
num_classes=20 # 20 classes in MSRAction3D
