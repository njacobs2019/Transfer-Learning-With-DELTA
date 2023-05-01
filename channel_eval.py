# -*- coding: utf-8 -*-
# from __future__ import print_function, division

# import os
# import sys
import torch
import torch.nn as nn
import torch.optim as optim

# import torch.nn.functional as F
# from torchvision import datasets, models, transforms
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
import time

# import argparse
import math
import json

# import pickle

# import numpy as np
# from torchnet import meter
# from PIL import ImageFile

from preprocessing import cifar10_datasets

# ImageFile.LOAD_TRUNCATED_IMAGES = True


## Parses command line arguments
# parser = argparse.ArgumentParser(description="DELTA")
# parser.add_argument("--data_dir")
# parser.add_argument("--channel_wei")
# parser.add_argument(
#     "--base_model",
#     choices=["resnet50", "resnet101", "inceptionv3"],
#     default="resnet101",
# )
# parser.add_argument(
#     "--base_task", choices=["imagenet", "places365"], default="imagenet"
# )
# parser.add_argument("--lr_init", type=float, default=0.01)

# args = parser.parse_args()
# print(args)

lr_init = 0.01

## *******

## Creates Datasets and DataLoaders
batch_size = 64

# Dictionary of "set_name":dataset_object (train, test)
image_datasets = cifar10_datasets()

set_names = list(image_datasets.keys())  # List of the dataset names

# Dataloaders
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
    )
    for x in set_names
}

dataset_sizes = {x: len(image_datasets[x]) for x in set_names}
class_names = image_datasets["train"].classes
num_classes = len(class_names)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "GPU is not running, using CPU"


## ***********

hook_layers = [
    "layer1.2.conv3",
    "layer2.3.conv3",
    "layer3.5.conv3",
    "layer4.2.conv3",
]

# model_target = resnet50(pretrained=True)
model_target = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

model_target.fc = nn.Linear(2048, num_classes)
model_target = model_target.to(device)


def train_classifier(model):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_init,
        momentum=0.9,
        weight_decay=1e-4,
    )
    num_epochs = 10
    decay_epochs = 6
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=math.exp(math.log(0.1) / decay_epochs)
    )
    since = time.time()
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            nstep = len(dataloaders[phase])
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if phase == "train" and i % 10 == 0:
                    corr_sum = torch.sum(preds == labels.data)
                    step_acc = corr_sum.double() / len(labels)
                    print(
                        "step: %d/%d, loss = %.4f, top1 = %.4f"
                        % (i, nstep, loss, step_acc)
                    )

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(
                "{} epoch: {:d} Loss: {:.4f} Acc: {:.4f}".format(
                    phase, epoch, epoch_loss, epoch_acc
                )
            )
            time_elapsed = time.time() - since
            print(
                "Training complete in {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )
            if epoch == num_epochs - 1:
                print(
                    "{} epoch: last Loss: {:.4f} Acc: {:.4f}".format(
                        phase, epoch_loss, epoch_acc
                    )
                )
        print()
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    return model


def feature_weight(model):
    filter_weight = []
    for i in range(len(hook_layers)):
        channel = model.state_dict()[hook_layers[i] + ".weight"].shape[0]
        layer_filter_weight = [0] * channel
        filter_weight.append(layer_filter_weight)

    criterion = nn.CrossEntropyLoss()
    model.eval()  # Set model to evaluate mode
    since = time.time()
    for i, (inputs, labels) in enumerate(dataloaders["train"]):
        if i >= 4:
            break
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss0 = criterion(outputs, labels)

        for name, module in model.named_modules():
            if not name in hook_layers:
                continue
            layer_id = hook_layers.index(name)
            channel = model.state_dict()[name + ".weight"].shape[0]
            for j in range(channel):
                tmp = model.state_dict()[name + ".weight"][j, :, :, :].clone()
                model.state_dict()[name + ".weight"][j, :, :, :] = 0
                # print(model.state_dict()[name + '.weight'][j,:,:,:])
                outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                diff = loss1 - loss0
                diff = diff.detach().cpu().numpy().item()
                hist = filter_weight[layer_id][j]
                filter_weight[layer_id][j] = 1.0 * (i * hist + diff) / (i + 1)
                print("%s:%d %.4f %.4f" % (name, j, diff, filter_weight[layer_id][j]))
                model.state_dict()[name + ".weight"][j, :, :, :] = tmp
                # print(model.state_dict()[name + '.weight'][j,:,:,:])
        print("step %d finished" % i)
        time_elapsed = time.time() - since
        print(
            "step Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
    return filter_weight


train_classifier(model_target)
filter_weight = feature_weight(model_target)
json.dump(filter_weight, open(args.channel_wei, "w"))
