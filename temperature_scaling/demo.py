#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : demo.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 2020/11/26 4:42 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 2020/11/26 4:42 下午 by shendu.ht  init
"""
import os

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import transforms

from models.densenet import DenseNet
from temperature_scaling import ModelWithTemperature


def demo(data, save, depth=40, growth_rate=12, batch_size=256):
    """
    Applies temperature scaling to a trained model.
    Takes a pretrained DenseNet-CIFAR100 model, and a validation set
    (parameterized by indices on train set).
    Applies temperature scaling, and saves a temperature scaled version.
    """

    # Load model state dict
    model_filename = os.path.join(save, 'model.pth')
    if not os.path.exists(model_filename):
        raise RuntimeError("Cannot find file %s to load" % model_filename)
    state_dict = torch.load(model_filename)

    # Load validation indices
    valid_indices_filename = os.path.join(save, 'valid_indices.pth')
    if not os.path.exists(valid_indices_filename):
        raise RuntimeError('Cannot find file %s to load' % valid_indices_filename)
    valid_indices = torch.load(valid_indices_filename)

    # Regenerate validation set loader
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=stdv), ])
    valid_set = datasets.CIFAR100(data, train=True, transform=test_transforms, download=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(valid_indices))

    # Load original model
    if (depth - 4) % 3:
        raise Exception("Invalid depth")
    block_config = [(depth - 4) // 6 for _ in range(3)]
    orig_model = DenseNet(growth_rate=growth_rate, block_config=block_config, num_classes=100).cuda()
    orig_model.load_state_dict(state_dict)

    # Now we're going to wrap the model with a decorator that adds temperature scaling
    model = ModelWithTemperature(orig_model)

    # Tune the model temperature, and save the results
    model.set_temperature(valid_loader)
    model_filename = os.path.join(save, "model_with_temperature.pth")
    torch.save(model.state_dict(), model_filename)
    print("Temperature scaled model sved to %s" % model_filename)
    print("Done!")
    return


if __name__ == '__main__':
    demo("input_path", "save_path")
