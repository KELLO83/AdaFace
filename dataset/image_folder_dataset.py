import os

import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import cv2

from .augmenter import Augmenter

def cv2_loader(path):
    """
    Custom loader using OpenCV to read an image and convert it to an RGB PIL Image.
    This is generally faster than the default PIL loader.
    """
    img = cv2.imread(path)
    # Convert from BGR (OpenCV default) to RGB (PIL/torchvision default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


class CustomImageFolderDataset(datasets.ImageFolder):

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=None,  # loader is now ignored, we use cv2_loader
                 is_valid_file=None,
                 low_res_augmentation_prob=0.0,
                 crop_augmentation_prob=0.0,
                 photometric_augmentation_prob=0.0,
                 swap_color_channel=False,
                 output_dir='./',
                 ):

        # We use our own faster loader, overriding the default
        super(CustomImageFolderDataset, self).__init__(root,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       loader=cv2_loader,
                                                       is_valid_file=is_valid_file)
        self.root = root
        self.augmenter = Augmenter(crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob)
        self.swap_color_channel = swap_color_channel
        self.output_dir = output_dir  # for checking the sanity of input images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # self.loader is cv2_loader, which returns an RGB PIL Image
        sample = self.loader(path)
        
        sample = Image.fromarray(np.asarray(sample)[:,:,::-1])

        sample = self.augmenter.augment(sample)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

