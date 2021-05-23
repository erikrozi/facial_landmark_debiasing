import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math

# TODO: Center crop? Horizontal flip? Rotations?

class RandomRotation(object):
    """Randomly rotate image
    
    Args:
        degrees (tuple): Degrees in range of random rotation, in tuple (min, max) or int
    """
    def __init__(self, degrees=(0, 360)):
        assert(isinstance(degrees, (tuple, int)))
        if isinstance(degrees, int):
            degrees = (-degrees, degrees)
        self.degrees = degrees
    
    def __call__(self, sample):
        '''
        Args:
            sample[0]: Tensor containing image
            sample[1]: array of dim [:, 2] containing landmark positions
        '''
        degrees = torch.rand(1)[0] * (self.degrees[1] - self.degrees[0]) + self.degrees[0]
        return Rotate(degrees)(sample)
    

class Rotate(object):
    """Rotate image in a sample with crop
    
    Args:
        degrees: Degrees to rotate image and landmarks by.
    """
    
    def __init__(self, degrees):
        self.degrees = degrees
        degrees = math.radians(degrees)
        self.rotation_matrix = torch.Tensor([[math.cos(degrees), - math.sin(degrees)], [math.sin(degrees), math.cos(degrees)]]).T

    def __call__(self, sample):
        '''
        Args:
            sample[0]: Tensor containing image
            sample[1]: array of dim [:, 2] containing landmark positions
        '''
        image, landmarks = sample
        h, w = image.shape[-2:]

        img =  transforms.RandomRotation((self.degrees, self.degrees))(image)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks[:, 0] -= h/2
        landmarks[:, 1] -= w/2
        
        landmarks = self.rotation_matrix @ torch.Tensor(landmarks).T
        landmarks = landmarks.T.numpy()
        landmarks[:, 0] += h/2
        landmarks[:, 1] += w/2
        

        return (img, landmarks)

class Rescale(object):
    """Rescale the image in a sample to a given size.
    
    Loosely Based on implementation from 
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, output_size dimensions are extended
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        '''
        Args:
            sample[0]: Tensor containing image
            sample[1]: array of dim [:, 2] containing landmark positions
        '''
        image, landmarks = sample
        h, w = image.shape[-2:]
        if isinstance(self.output_size, int):
            new_h = self.output_size
            new_w = self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img =  transforms.Resize((new_h, new_w))(image)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]
        

        return (img, landmarks)
    
class NormalizeLandmarks(object):
    '''Normalize landmarks between 0 and 1'''
    
    def __init__(self):
        pass
    
    def __call__(self, sample):
        '''
        Args:
            sample[0]: Tensor containing image
            sample[1]: array of dim [:, 2] containing landmark positions
        '''
        image, landmarks = sample
        
        h, w = image.shape[-2:]

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [1 / w, 1 / h]

        return (image, landmarks)
    