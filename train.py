import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from tqdm import tqdm

from preprocessing import cifar10_datasets


def main(name, lr_init=0.01, alpha=0.01, beta=0.01, num_epochs=14, batch_size=128):
    """
    This function trains a model with the input hyperparameters and saves it
    """

    # Program parameters
    lr_scheduler = "explr"  # "steplr", "explr"
    reg_type = "att_fea_map"  # "l2, "l2_sp", "fea_map", "att_fea_map"
    channel_wei = "./config/channel_wei.cifar10.json"

    # Create Datasets
    image_datasets = (
        cifar10_datasets()
    )  # Dictionary of "set_name":dataset_object (train, test)

    set_names = list(image_datasets.keys())  # List of the dataset names
    dataset_sizes = {x: len(image_datasets[x]) for x in set_names}
    num_classes = 10

    # Dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
        )
        for x in set_names
    }

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "GPU is not running, using CPU"

    # Creating the source model
    model_source = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model_source.to(device)
    for param in model_source.parameters():
        param.requires_grad = False
    model_source.eval()

    model_source_weights = {}
    for name, param in model_source.named_parameters():
        model_source_weights[name] = param.detach()

    # Creating the target model
    model_target = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model_target.fc = nn.Linear(2048, num_classes)
    model_target.to(device)

    # Getting channel weights
    channel_weights = []
    if reg_type == "att_fea_map" and channel_wei:
        for js in json.load(open(channel_wei)):
            js = np.array(js)
            js = (js - np.mean(js)) / np.std(js)
            cw = torch.from_numpy(js).float().to(device)
            cw = F.softmax(cw / 5, dim=0).detach()
            channel_weights.append(cw)

    layer_outputs_source = []
    layer_outputs_target = []

    def for_hook_source(module, input, output):
        layer_outputs_source.append(output)

    def for_hook_target(module, input, output):
        layer_outputs_target.append(output)

    fc_name = "fc."

    hook_layers = [
        "layer1.2.conv3",
        "layer2.3.conv3",
        "layer3.5.conv3",
        "layer4.2.conv3",
    ]

    def register_hook(model, func):
        for name, layer in model.named_modules():
            if name in hook_layers:
                layer.register_forward_hook(func)

    register_hook(model_source, for_hook_source)
    register_hook(model_target, for_hook_target)

    def reg_classifier(model):
        l2_cls = torch.tensor(0.0).to(device)
        for name, param in model.named_parameters():
            if name.startswith(fc_name):
                l2_cls += 0.5 * torch.norm(param) ** 2
        return l2_cls

    def reg_l2sp(model):
        fea_loss = torch.tensor(0.0).to(device)
        for name, param in model.named_parameters():
            if not name.startswith(fc_name):
                fea_loss += 0.5 * torch.norm(param - model_source_weights[name]) ** 2
        return fea_loss

    def reg_fea_map(inputs):
        _ = model_source(inputs)
        fea_loss = torch.tensor(0.0).to(device)
        for fm_src, fm_tgt in zip(layer_outputs_source, layer_outputs_target):
            b, c, h, w = fm_src.shape
            fea_loss += 0.5 * (torch.norm(fm_tgt - fm_src.detach()) ** 2)
        return fea_loss

    def flatten_outputs(fea):
        return torch.reshape(
            fea, (fea.shape[0], fea.shape[1], fea.shape[2] * fea.shape[3])
        )

    def reg_att_fea_map(inputs):
        _ = model_source(inputs)
        fea_loss = torch.tensor(0.0).to(device)
        for i, (fm_src, fm_tgt) in enumerate(
            zip(layer_outputs_source, layer_outputs_target)
        ):
            b, c, h, w = fm_src.shape
            fm_src = flatten_outputs(fm_src)
            fm_tgt = flatten_outputs(fm_tgt)
            div_norm = h * w
            distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)
            distance = c * torch.mul(channel_weights[i], distance**2) / (h * w)
            fea_loss += 0.5 * torch.sum(distance)
        return fea_loss

    def train_model(model, criterion, optimizer, scheduler, num_epochs):
        # TensorBoard
        writer = SummaryWriter(name)

        for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch", position=0):
            for phase in ["train", "test"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(
                    dataloaders[phase],
                    desc=(phase + "ing batches"),
                    unit="batch",
                    position=1,
                    leave=False,
                ):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss_main = criterion(outputs, labels)
                        loss_classifier = 0
                        loss_feature = 0
                        if not reg_type == "l2":
                            loss_classifier = reg_classifier(model)
                        if reg_type == "l2_sp":
                            loss_feature = reg_l2sp(model)
                        elif reg_type == "fea_map":
                            loss_feature = reg_fea_map(inputs)
                        elif reg_type == "att_fea_map":
                            loss_feature = reg_att_fea_map(inputs)
                        loss = loss_main + alpha * loss_feature + beta * loss_classifier

                        _, preds = torch.max(outputs, 1)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    layer_outputs_source.clear()
                    layer_outputs_target.clear()

                if phase == "train":
                    writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                # Cache the metrics for logging
                if epoch == num_epochs - 1:
                    if phase == "train":
                        train_metrics = (epoch_loss, epoch_acc)
                    else:
                        test_metrics = (epoch_loss, epoch_acc)

                # Record the loss and accuracy with TensorBoard
                writer.add_scalar("Loss/" + phase, epoch_loss, epoch)
                writer.add_scalar("Accuracy/" + phase, epoch_acc, epoch)

                if phase == "train" and abs(epoch_loss) > 1e8:
                    break

        writer.add_hparams(
            {
                "lr_init": lr_init,
                "alpha": alpha,
                "beta": beta,
                "num_epochs": num_epochs,
            },
            {
                "hparam/test_accuracy": test_metrics[1],
                "hparam/test_loss": test_metrics[0],
                "hparam/train_accuracy": train_metrics[1],
                "hparam/train_loss": train_metrics[0],
            },
        )

        writer.close()
        return model

    # Optimizer
    if reg_type == "l2":
        optimizer_ft = optim.SGD(
            filter(lambda p: p.requires_grad, model_target.parameters()),
            lr=lr_init,
            momentum=0.9,
            weight_decay=1e-4,
        )
    else:
        optimizer_ft = optim.SGD(
            filter(lambda p: p.requires_grad, model_target.parameters()),
            lr=lr_init,
            momentum=0.9,
        )

    # Learning Rate Scheduler
    decay_epochs = int(0.67 * num_epochs) + 1
    if lr_scheduler == "steplr":
        lr_decay = optim.lr_scheduler.StepLR(
            optimizer_ft, step_size=decay_epochs, gamma=0.1
        )
    elif lr_scheduler == "explr":
        lr_decay = optim.lr_scheduler.ExponentialLR(
            optimizer_ft, gamma=math.exp(math.log(0.1) / decay_epochs)
        )

    criterion = nn.CrossEntropyLoss()
    train_model(model_target, criterion, optimizer_ft, lr_decay, num_epochs)


if __name__ == "__main__":
    main(name="50_epoch_run", num_epochs=50)
