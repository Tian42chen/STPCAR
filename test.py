import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time

from models import P4Transformer
from data import MSRAction3D, HOI4D
from utils import accuracy
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model, name):
    print('Loading model...')
    model.load_state_dict(torch.load(config.save_model_path + name))

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

        log='Accuracy of the network on the {} test clips: {:.2f}%\n'.format(total, 100 * correct / total)
        print(log, end='')
        with open(f'{config.log_path}log.txt', 'a') as f:
            f.write(log)
        
    print('Finished testing in {:.2f} seconds'.format(time.time() - start_time))


def load_and_test(load_name, model, test_loader):
    # Test model
    print('Test model: ', load_name)
    load_model(model, load_name)
    test(model, test_loader)

def main():
    # Parameters
    assert(device != 'cpu'), 'No GPU available'

    # Load test set
    print('Loading data...')
    test_set = MSRAction3D(config.data_path, train=False)
    # test_set = HOI4D(config.data_path, train=False)

    print(f'Number of test clips: {len(test_set)}')

    print('Creating data loader...')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Load model
    model = P4Transformer(
        radius=config.radius, nsamples=config.nsamples, spatial_stride=config.spatial_stride, # P4DConv: spatial
        temporal_kernel_size=config.temporal_kernel_size, temporal_stride=config.temporal_stride, # P4DConv: temporal
        emb_complex=config.emb_complex,  # embedding: relu
        dim=config.dim, depth=config.depth, heads=config.heads, dim_head=config.dim_head,  # transformer
        mlp_dim=config.mlp_dim, num_classes=config.num_classes # output
    ).to(device)

    load_and_test('P4T-epoch20.pth', model, test_loader)
    load_and_test('P4T-epoch30.pth', model, test_loader)
    load_and_test('P4T-epoch50.pth', model, test_loader)

if __name__ == '__main__':
    main()