#!/usr/bin/env python

"""A simple python script template.
"""
from __future__ import print_function
from PIL import Image
from torch.autograd import Variable
from train import load_pre_trained_model
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import time
import copy
import numpy as np
import json

# Process a PIL image for use in a PyTorch model
# https://discuss.pytorch.org/t/how-to-classify-single-image-using-loaded-net/1411  
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    pil_image = Image.open(image)
    image_tensor = data_transform(pil_image).float()   
    image_tensor = image_tensor.unsqueeze(0) # this is for VGG, ResNet 
    image = Variable(torch.FloatTensor(image_tensor), requires_grad=True)            
    return image
	
def predict(input, checkpoint, category_names='cat_to_name.json', top_k=1, gpu=True):
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint)
    model = load_pre_trained_model(arch=checkpoint['arch'], 
                                        output_size=len(checkpoint['class_to_idx']), 
                                        hidden_units=checkpoint['hidden_units'])

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    model = model.to(device)       
    model.eval()
     
    image = process_image(input)
       
    if gpu and torch.cuda.is_available():
        image = image.cuda()

    outputs = model(image).topk(top_k)

    if gpu and torch.cuda.is_available():
        probs = F.softmax(outputs[0].data, dim=1).cpu().numpy()[0]
        classes = outputs[1].data.cpu().numpy()[0]
    else:       
        probs = F.softmax(outputs[0].data, dim=1).numpy()[0]
        classes = outputs[1].data.numpy()[0]

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
   
    classes_names = [cat_to_name[str(x)] for x in classes]
        
    if top_k == 1:
        print('It''s a {}({}) with a associated probability of {:.2%}'.format(classes_names[0], classes[0], probs[0]))

    return probs, classes, classes_names
	
def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=str, help='input image to predict')
    parser.add_argument('--checkpoint', type=str, help='load saved checkpoint file')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='json file containing mapping of categories to real names')    
    parser.add_argument('--top_k', type=int, default=1, help='top K classes')
    parser.add_argument('--gpu', action='store_true', default=True, help='use GPU if available')
    args = parser.parse_args(arguments)	
    predict(args.input, args.checkpoint, args.category_names, args.top_k, args.gpu)
	
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))