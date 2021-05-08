import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch

class CelebaDataset(Dataset):
    ''' CelebA Dataset '''

    def __init__(self, landmark_file, attribute_file, image_dir, transform=None, landmark_transform=None):
        '''
        Args:
            landmark_file (string): Path to landmark file with landmark annotations
            attribute_file (string): Path to attribute file with class annotations
            image_dir (string): Path to image directory
            transform: Optional transform to be applied on image only
            landmark_transform: Custom transforms to be applied on image and landmark
        '''
        self.landmarks_frame = pd.read_csv(landmark_file)
        self.attributes_frame = pd.read_csv(attribute_file)
        self.image_dir = image_dir
        self.transform = transform
        self.landmark_transform = landmark_transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, 
                                self.landmarks_frame.iloc[idx, 0])

        image = Image.open(img_name)
        width, height = image.size
        
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        
        # Scales landmarks between 0 and 1. Temporary solution
        #landmarks[:, 0] /= width
        #landmarks[:, 1] /= height
        
        attributes = self.attributes_frame.iloc[idx, 1:]
        attributes = np.array([attributes]).astype('int')
        attributes = torch.as_tensor(attributes, dtype=torch.int)

        if self.transform:
            image = self.transform(image)
        if self.landmark_transform:
            image, landmarks = self.landmark_transform((image, landmarks))

        landmarks = torch.as_tensor(landmarks, dtype=torch.float32)

        sample = {'image': image, 'landmarks': landmarks, 'attributes': attributes}

        return sample
    
    
class WFLWDataset(Dataset):
    ''' WFLW Dataset '''

    def __init__(self, annotation_file, image_dir, transform=None, landmark_transform=None, crop=True):
        '''
        Args:
            annotation_file (string): Path to annotation file
            image_dir (string): Path to image directory
            transform: Optional transform to be applied on image only
            landmark_transform: Custom transforms to be applied on image and landmark
            crop: Crop face to bounding box
        '''
        self.annotations_frame =pd.read_csv(annotation_file, sep=' ', header=None)
        self.image_dir = image_dir
        self.transform = transform
        self.landmark_transform = landmark_transform
        self.crop = crop

    def __len__(self):
        return len(self.annotations_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, 
                                self.annotations_frame.iloc[idx, -1])

        image = Image.open(img_name)
        width, height = image.size
        
        landmarks = self.annotations_frame.iloc[idx, 0:196]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        
        crop_coord = np.array(self.annotations_frame.iloc[idx, 196:200])
        if self.crop:
            image = image.crop((crop_coord[0], crop_coord[1], crop_coord[2], crop_coord[3]))
            landmarks[:, 0] -= crop_coord[0]
            landmarks[:, 1] -= crop_coord[1]
        
        # Scales landmarks between 0 and 1. Temporary solution
        #landmarks[:, 0] /= width
        #landmarks[:, 1] /= height
        
        attributes = self.annotations_frame.iloc[idx, 200:206]
        attributes = np.array([attributes]).astype('int')
        attributes = torch.as_tensor(attributes, dtype=torch.int)

        if self.transform:
            image = self.transform(image)
        if self.landmark_transform:
            image, landmarks = self.landmark_transform((image, landmarks))

        landmarks = torch.as_tensor(landmarks, dtype=torch.float32)

        sample = {'image': image, 'landmarks': landmarks, 'attributes': attributes}

        return sample