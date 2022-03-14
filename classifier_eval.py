import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import argparse

from classifier_network import Resnet18Network, SqueezeNet

def eval_folder(data_dir, mode, model_dir):

    with torch.no_grad():
        num_classes = len(os.listdir(os.path.join(data_dir, mode)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model = Resnet18Network(num_classes)
        model = SqueezeNet(num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model.eval()

        # Iterate over data.
        for inputs, labels in tqdm(dataloaders[args.mode]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # print('Labels: ', labels.data.tolist())
            # print('Predictions: ', preds.tolist())

            llabels = labels.data.tolist()
            lpreds = preds.tolist()
            
            for i in range(len(llabels)):
                key = class_names[llabels[i]]
                stats[key][1] += 1
                if preds[i] == llabels[i]:
                    stats[key][0] += 1
        print("Stats: ", stats)

if __name__ == "__main__":
    # Arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--model_dir", help="path to checkpoint", type=str, required=True)
    parser.add_argument("-m", "--mode", help="Train/Val dataset evaluation mode", type=str, required=True, default='val')
    parser.add_argument("-d", "--data_dir", help="Path to data folder", type=str, default='data')
    args = parser.parse_args()

    # Load data

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = args.data_dir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in [args.mode]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in [args.mode]}
    dataset_sizes = {x: len(image_datasets[x]) for x in [args.mode]}
    class_names = image_datasets[args.mode].classes

    stats = {key: [0, 0] for key in class_names}
    eval_folder(args.data_dir, args.mode, args.model_dir)