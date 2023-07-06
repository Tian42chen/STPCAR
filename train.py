import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time

from models import P4Transformer
from data import MSRAction3D
from utils import accuracy, MetricLogger
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model, name):
    print('Saving model...')
    torch.save(model.state_dict(), config.save_model_path + name)
    
def load_model(model, name):
    print('Loading model...')
    model.load_state_dict(torch.load(config.save_model_path + name))

def train(model, train_loader, criterion, optimizer):
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
            
            # Calculate accuracy
            acc1=accuracy(outputs, labels, topk=(1,))

            # Print statistics
            running_loss += loss.item()
            running_acc += acc1[0].item()
            
            if (i+1) % config.print_interval == 0:
                print (
                    'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch+1, config.num_epochs, i+1, len(train_loader), running_loss/config.print_interval, running_acc/config.print_interval)
                )
                running_loss = 0.0
                running_acc = 0.0
        
        save_model(model, f"P4T-epoch{epoch+1}.pth")

    print('Finished training in {:.2f} seconds'.format(time.time() - start_time))

def test(model, test_loader):
    print('Testing...')
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for clip, labels in test_loader:
            clip = clip.to(device)
            labels = labels.to(device)
            outputs = model(clip)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the test clips: {100 * correct / total} %')
    print('Finished testing in {:.2f} seconds'.format(time.time() - start_time))

def main():
    # Parameters
    assert(device != 'cpu'), 'No GPU available'

    # Load data
    print('Loading data...')
    train_data = MSRAction3D(config.data_path, train=True)
    test_data = MSRAction3D(config.data_path, train=False)

    print('Creating data loaders...')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    print(f'Number of training clips: {len(train_data)}')
    print(f'Number of test clips: {len(test_data)}')

    # Model
    print('Creating model...')
    model = P4Transformer(
        radius=config.radius, nsamples=config.nsamples, spatial_stride=config.spatial_stride, # P4DConv: spatial
        temporal_kernel_size=config.temporal_kernel_size, temporal_stride=config.temporal_stride, # P4DConv: temporal
        emb_relu=config.emb_relu,  # embedding: relu
        dim=config.dim, depth=config.depth, heads=config.heads, dim_head=config.dim_head,  # transformer
        mlp_dim=config.mlp_dim, num_classes=config.num_classes # output
    ).to(device)

    print('Creating optimizer...')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

    # Train
    train(model, train_loader, criterion, optimizer)

    # Test
    test(model, test_loader)


if __name__ == '__main__':
    main()