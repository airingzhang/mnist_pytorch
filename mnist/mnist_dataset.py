'''
Created on Feb 25, 2019

@author: airingzhang
'''
import torch.utils.data as data
import os
import numpy as np
import torch
from utils import process_image_file, process_label_file

class MNIST(data.Dataset):
    """
     Args:
        pt files path (string): include images and labels
        train (bool, optional): If True, sepcial augmentation may be added to
        
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
            
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.    
    """

    def __init__(self, data_file, train=True, stats_file=None):
        self.data_file = data_file
        self.train = train  # training set or test set
        self.data, self.targets = torch.load(data_file)
        if stats_file:
            self.mean, self.std = np.load(stats_file)
        else:
            self.mean, self.std = 0., 1.
    
    def transform_img(self, img):
        img = img.float().div(255)
        if self.mean != 0:
            img.sub_(self.mean)
        if self.std != 1:
            img.div_(self.std)
        img.unsqueeze_(0)
        return img
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            # do something in the future to differentiate behaviors during training phase and testing phase 
            pass 
        
        img, target = self.data[index], int(self.targets[index])
        img = self.transform_img(img)

        return img, target

    def __len__(self):
        return len(self.data)
        
    @staticmethod    
    def preprocessing(basedir, split_train=True, split_ratio=0.1):
        """
        Pre-processing raw data files.
        One should download the MNIST data beforehand using the bash scripts in `run_all.sh`
        If no data is found, prompt error info.
        
        Args:
            basedir (str): folder where the raw data files located 
            split_train: if True split training set into training set and validation set
            ratio (float): ratio of the validation set 
        """
        train_file_set = {'image': 'train-images-idx3-ubyte', 
                      'label': 'train-labels-idx1-ubyte'}
        test_file_set = {'image': 't10k-images-idx3-ubyte',
                         'label': 't10k-labels-idx1-ubyte'}
        training_file = 'training.pt'
        test_file = 'testing.pt'
        # process and save as torch files
        print('Processing...')
        for val in train_file_set.values():
            if not os.path.exists(os.path.join(basedir, val)):
                print( '%s does not exist. Check the dataset folder.' % os.path.join(basedir, val))
                return 
            
        for val in test_file_set.values():
            if not os.path.exists(os.path.join(basedir, val)):
                print( '%s does not exist. Check the dataset folder.' % os.path.join(basedir, val))
                return 
            
        length, labels = process_label_file(os.path.join(basedir, train_file_set['label']))
        train_labels_pt = torch.from_numpy(labels).view(length).long()
        
        length, num_rows, num_cols, images = process_image_file(os.path.join(basedir, train_file_set['image']))
        train_image_pt = torch.from_numpy(images).view(length, num_rows, num_cols)
        
        mean_train, std_train = np.mean(images)/255.0, np.std(images)/255.0
        length, labels = process_label_file(os.path.join(basedir, test_file_set['label']))
        test_labels_pt = torch.from_numpy(labels).view(length).long()
        
        length, num_rows, num_cols, images = process_image_file(os.path.join(basedir, test_file_set['image']))
        test_image_pt = torch.from_numpy(images).view(length, num_rows, num_cols)
        
        
        np.save(os.path.join(basedir, 'stats'), [mean_train, std_train] )
        
        with open(os.path.join(basedir, training_file), 'wb') as f:
            torch.save((train_image_pt, train_labels_pt), f)
        with open(os.path.join(basedir, test_file), 'wb') as f:
            torch.save((test_image_pt, test_labels_pt), f)
            
        if split_train:
            print('Spliting training set...')
            idx = np.random.permutation(len(train_labels_pt))
            val_len = int(len(train_labels_pt)* split_ratio)
            label_validation_split = train_labels_pt[idx[: val_len]]
            label_train_split = train_labels_pt[idx[val_len:]]
            image_validation_split = train_image_pt[idx[: val_len]]
            image_train_split = train_image_pt[idx[val_len:]]
            with open(os.path.join(basedir, 'validation_split.pt'), 'wb') as f:
                torch.save((image_validation_split, label_validation_split), f)
            with open(os.path.join(basedir, 'training_split.pt'), 'wb') as f:
                torch.save((image_train_split, label_train_split), f)

        print('Done!')
    
