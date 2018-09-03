#!/usr/bin/env python

"""A simple python script template.
"""
from __future__ import print_function
from collections import OrderedDict
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import copy

def get_data_transforms():
	# Define your transforms for the training, validation sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms

def load_pre_trained_model(arch, output_size, hidden_units):
    pre_trained_models = {'densenet': models.densenet121(pretrained=True), 'alexnet': models.alexnet(pretrained=True), 'vgg': models.vgg19(pretrained=True)}
    model = pre_trained_models[arch]

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # https://medium.com/@14prakash/almost-any-image-classification-problem-using-pytorch-i-am-in-love-with-pytorch-26c7aa979ec4
    if arch == 'densenet':
        num_ftrs = model.classifier.in_features               
        classifier = nn.Sequential(OrderedDict([
                                    ('drop1',nn.Dropout(0.5)),	
                                    ('fc1', nn.Linear(num_ftrs, hidden_units)),
                                    ('relu', nn.ReLU()),
                                    ('fc2', nn.Linear(hidden_units, output_size))
                                    ]))
        model.classifier = classifier
    else:
        # Features, removing the last layer
        features = list(model.classifier.children())[:-1]
    
        # Number of filters in the bottleneck layer
        num_filters = model.classifier[len(features)].in_features

        # Extend the existing architecture with new layers
        features.extend([
            nn.Dropout(0.5),
            nn.Linear(num_filters, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_size),
        ])
    
        model.classifier = nn.Sequential(*features)
            
    return model    

# This method trains a model 
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(data_dir, arch, hidden_units, lr=0.001, num_epochs=25, gpu=True, save_dir='checkpoint.pt'):
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x), transform=get_data_transforms()[x]) for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid']}    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    output_size = len(image_datasets['train'].classes)

    model = load_pre_trained_model(arch, output_size, hidden_units)
    model = model.to(device)

    # define optimizer and criterion       
    criterion = nn.CrossEntropyLoss()
    if arch == 'densenet':
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr) 
        
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:                
                inputs = inputs.to(device)
                labels = labels.to(device)               

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward, track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Store class_to_idx into a model property
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    checkpoint_dict = {
        'arch': arch,
        'class_to_idx': model.class_to_idx, 
        'state_dict': model.state_dict(),
        'hidden_units': hidden_units
    }
             
    torch.save(checkpoint_dict, save_dir)
    return model	
	
def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='path to the image dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='path to directory to save checkpoints')	
    parser.add_argument('--arch', type=str, default='vgg', help='chosen model')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--hidden_units', default=512, type=int, help='number of hidden units')
    parser.add_argument('--gpu', action='store_true', default=True, help='use GPU if available')	
    args = parser.parse_args(arguments)	
    train_model(args.data_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu, args.save_dir)
       
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))