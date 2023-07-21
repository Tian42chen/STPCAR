import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time

from models import P4Transformer
from data import MSRAction3D, HOI4D
from utils import accuracy, WarmupMultiStepLR
from test import test

import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model, name):
    print('Saving model...')
    torch.save(model.state_dict(), config.save_model_path + name)

def train(model, train_loader, test_loader, criterion, optimizer, lr_scheduler):
    print('Training...')
    start_time = time.time()
    model.train()
    for epoch in range(config.num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        for i, (clip, labels) in enumerate(train_loader):
            clip = clip.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(clip)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Calculate accuracy
            acc1=accuracy(outputs, labels, topk=(1,))

            # Print statistics
            running_loss += loss.item()
            running_acc += acc1[0].item()
            
            if (i+1) % config.print_interval == 0:
                log='Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%\n'.format(epoch+1, config.num_epochs, i+1, len(train_loader), running_loss/config.print_interval, running_acc/config.print_interval)
                print (log, end='')
                with open('log.txt', 'a') as f:
                    f.write(log)
                running_loss = 0.0
                running_acc = 0.0
        
        
        save_model(model, f"P4T-epoch{epoch+1}.pth")
        test(model, test_loader)

    print('Finished training in {:.2f} seconds'.format(time.time() - start_time))

def main():
    # Parameters
    assert(device != 'cpu'), 'No GPU available'

    # Load data
    print('Loading data...')
    train_set = MSRAction3D(config.data_path, train=True)
    test_set = MSRAction3D(config.data_path, train=False)
    # train_set = HOI4D(config.data_path, train=True)
    # test_set = HOI4D(config.data_path, train=False)

    print(f'Number of training clips: {len(train_set)}')
    print(f'Number of test clips: {len(test_set)}')

    print('Creating data loader...')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.workers)

    # Model
    print('Creating model...')
    model = P4Transformer(
        radius=config.radius, nsamples=config.nsamples, spatial_stride=config.spatial_stride, # P4DConv: spatial
        temporal_kernel_size=config.temporal_kernel_size, temporal_stride=config.temporal_stride, # P4DConv: temporal
        emb_complex=config.emb_complex,  # embedding: relu
        dim=config.dim, depth=config.depth, heads=config.heads, dim_head=config.dim_head,  # transformer
        mlp_dim=config.mlp_dim, num_classes=config.num_classes # output
    ).to(device)

    print(
        f"Total number of paramerters in networks is {sum(x.numel() for x in model.parameters())}  "
    )

    print('Creating optimizer...')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)


    print('Creating learning rate scheduler...')
    warmup_iters = config.lr_warmup_epochs * len(train_loader)
    lr_milestones=[(len(train_loader) * epoch) for epoch in config.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=config.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

    # Train
    train(model, train_loader, test_loader, criterion, optimizer, lr_scheduler)


if __name__ == '__main__':
    main()