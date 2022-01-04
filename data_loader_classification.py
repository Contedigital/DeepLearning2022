import torch
import pandas as pd
import numpy as np
import os
import re 
import random
import glob
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.io import read_image


def prepare_labels(img_path, label_file):
    # extract file name of the image from path
    x = re.search(r"ISIC_\d+", img_path)
    img_file_name = x.group()

    df_labels = label_file.loc[label_file.image == img_file_name, :].drop("image", axis=1)
    target_index = list(df_labels.values[0]).index(1)
    target_index =  torch.tensor(target_index).type(torch.long)
    return target_index


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, noisy_transform=None, test=False):
        self.img_dir = img_dir
        self.label_file = label_file
        self.transform = transform
        self.noisy_transform = noisy_transform
        self.test = test

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx]
        image = read_image(img_path)
        target_index = prepare_labels(img_path, self.label_file)
        if self.transform:
            normal_image = self.transform(image)
        # the test set does not require the noise
        if self.test:
            return normal_image, target_index
        else:
            if self.noisy_transform:
                noisy_image = self.noisy_transform(image)
            return normal_image, noisy_image, target_index


def get_train_valid_loader(data_dir,
                           label_path,
                           batch_size,
                           random_seed,
                           transform,
                           noisy_transform,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - noisy_transform: transformation pipeline for generating the noisy data set
    - transform: transformation pipeline for the original/target data set
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # list of image filenames
    img_dir = [name for name in glob.glob(f'{data_dir}/*.jpg')]

    # file containing all the labels
    label_file = pd.read_csv(label_path)
    # get number of classes
    n_classes = len(list(label_file.drop("image", axis=1).columns))
    
    # load the dataset
    full_dataset = CustomImageDataset(img_dir=img_dir, label_file=label_file, transform=transform,  noisy_transform=noisy_transform)
    
    #split into train/validation
    train_size = int((1-valid_size) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, val_size], generator = torch.Generator().manual_seed(42) )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, #sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=shuffle, #sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:

        sample_loader = DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, noisy_images, labels = next(data_iter)

        plt.figure()
        #subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(2,1) 

        axarr[0].imshow(np.transpose(images[0], (1, 2, 0)))
        axarr[1].imshow(np.transpose(noisy_images[0], (1, 2, 0)))

    return (train_loader, valid_loader, n_classes)


def get_test_loader(data_dir,
                    label_path,
                    batch_size,
                    transform,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # list of image filenames
    img_dir = [name for name in glob.glob(f'{data_dir}/*.jpg')]

    # file containing all the labels
    label_file = pd.read_csv(label_path)

    # load the dataset
    test_dataset = CustomImageDataset(img_dir=img_dir, label_file=label_file, transform=transform, test=True)

    valid_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return valid_loader