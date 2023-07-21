# Hyperparameters
learning_rate = 0.01
momentum = 0.9
weight_decay = 1e-4
batch_size = 16
# num_epochs = 30
num_epochs = 50
print_pre_epoch = 10
workers=4

# lr_milestones=[12, 18]
lr_milestones=[20, 30]
lr_gamma=0.1
lr_warmup_epochs=10

# Data paths
data_path = './data/msr/testpcd/'
# data_path = './data/hoi4d/handposePcd/'

# Model paths
save_model_path = './checkpoints/'

# log paths
log_path = './logs/'

# Model parameters
## P4DConv: spatial
radius=0.7
nsamples=32
spatial_stride=32
## P4DConv: temporal
temporal_kernel_size=3
temporal_stride=2
## embedding: relu
emb_complex=False
# Transformer
dim=1024
depth=5
heads=8
dim_head=128
## output
mlp_dim=2048
