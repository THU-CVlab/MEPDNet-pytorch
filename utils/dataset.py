import os
import glob
import torch
import numpy as np

from PIL import Image
from os import listdir
from os.path import join,split,splitext
from utils import is_image_file
from torch.utils.data import Dataset


class SingleDataset(Dataset):
    def __init__(self, img_dir, mask_dir, input_transform, target_transform, logger=None):
        self.img_dir = np.load(img_dir)
        self.mask_dir = mask_dir
        self.img_file_names = [join('data/imgs', file_name) for file_name in self.img_dir if is_image_file(file_name)]
        self.mask_file_names = [join(mask_dir, file_name.replace('\\', '/').split('/')[-1].replace('.jpg', '.npy')) for file_name in self.img_dir if is_image_file(file_name)]
        if logger is not None:
            logger.info(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        else:
            print(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self,i):
        img = np.array(Image.open(self.img_file_names[i]))
        mask = np.load(self.mask_file_names[i])

        pad = lambda img: np.stack([img,img,img],axis=2) if (len(img.shape)==2 or img.shape[2]==1) else img
        img = pad(img)
        img = self.input_transform(img)
        mask = self.target_transform(mask)
        return img, mask

class SeqDataset(Dataset):
    def __init__(self, img_dir, mask_dir, input_transform, target_transform, logger=None):
        self.img_dir = np.load(img_dir)
        self.mask_dir = mask_dir
        self.img_file_names = [join('data/imgs', file_name) for file_name in self.img_dir if is_image_file(file_name)]
        self.mask_file_names = [join(mask_dir, file_name.replace('\\', '/').split('/')[-1].replace('.jpg', '.npy')) for file_name in self.img_dir if is_image_file(file_name)]
        if logger is not None:
            logger.info(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        else:
            print(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self, i):
        if i % 30 == 0:
            imgu1 = np.array(Image.open(self.img_file_names[i]))
        else:
            imgu1 = np.array(Image.open(self.img_file_names[i - 1]))
        img = np.array(Image.open(self.img_file_names[i]))
        if i % 30 == 29:
            imgd1 = np.array(Image.open(self.img_file_names[i]))
        else:
            imgd1 = np.array(Image.open(self.img_file_names[i + 1]))
        mask = np.load(self.mask_file_names[i])

        pad = lambda img: np.stack([img,img,img],axis=2) if (len(img.shape)==2 or img.shape[2]==1) else img
        imgu1, img, imgd1 = pad(imgu1), pad(img), pad(imgd1)

        imgu1, img, imgd1 = self.input_transform(imgu1), self.input_transform(img), self.input_transform(imgd1)
        mask = self.target_transform(mask)

        return imgu1, img, imgd1, mask

class SingleTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, input_transform, target_transform, logger=None):
        self.img_dir = np.load(img_dir)
        self.mask_dir = mask_dir
        self.img_file_names = [join('data/imgs', file_name) for file_name in self.img_dir if is_image_file(file_name)]
        self.mask_file_names = [join(mask_dir, file_name.replace('\\', '/').split('/')[-1].replace('.jpg', '.npy')) for file_name in self.img_dir if is_image_file(file_name)]
        if logger is not None:
            logger.info(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        else:
            print(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self,i):
        img = np.array(Image.open(self.img_file_names[i]))
        mask = np.load(self.mask_file_names[i])

        pad = lambda img: np.stack([img,img,img],axis=2) if (len(img.shape)==2 or img.shape[2]==1) else img
        img = pad(img)
        img = self.input_transform(img)
        mask = self.target_transform(mask)
        return img, mask, self.img_file_names[i]


class SeqTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, input_transform, target_transform, logger=None):
        self.img_dir = np.load(img_dir)
        self.mask_dir = mask_dir
        self.img_file_names = [join('data/imgs', file_name) for file_name in self.img_dir if is_image_file(file_name)]
        self.mask_file_names = [join(mask_dir, file_name.replace('\\', '/').split('/')[-1].replace('.jpg', '.npy')) for file_name in self.img_dir if is_image_file(file_name)]
        if logger is not None:
            logger.info(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        else:
            print(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self,i):
        if i % 30 == 0:
            imgu1 = np.array(Image.open(self.img_file_names[i]))
        else:
            imgu1 = np.array(Image.open(self.img_file_names[i - 1]))
        img = np.array(Image.open(self.img_file_names[i]))
        if i % 30 == 29:
            imgd1 = np.array(Image.open(self.img_file_names[i]))
        else:
            imgd1 = np.array(Image.open(self.img_file_names[i + 1]))
        mask = np.load(self.mask_file_names[i])

        pad = lambda img: np.stack([img,img,img],axis=2) if (len(img.shape)==2 or img.shape[2]==1) else img
        imgu1, img, imgd1 = pad(imgu1), pad(img), pad(imgd1)

        imgu1, img, imgd1 = self.input_transform(imgu1), self.input_transform(img), self.input_transform(imgd1)
        mask = self.target_transform(mask)

        return imgu1, img, imgd1, mask, self.img_file_names[i]


class SingleUseDataset(Dataset):
    def __init__(self, img_dir, input_transform, logger=None):
        self.img_dir = np.load(img_dir)
        self.img_file_names = [join('data/imgs', file_name) for file_name in self.img_dir if is_image_file(file_name)]
        if logger is not None:
            logger.info(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        else:
            print(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        self.input_transform = input_transform

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self,i):
        img = np.array(Image.open(self.img_file_names[i]))

        pad = lambda img: np.stack([img,img,img],axis=2) if (len(img.shape)==2 or img.shape[2]==1) else img
        img = pad(img)
        img = self.input_transform(img)
        return img, self.img_file_names[i]


class SeqUseDataset(Dataset):
    def __init__(self, img_dir, input_transform, logger=None):
        self.img_dir = np.load(img_dir)
        self.img_file_names = [join('data/imgs', file_name) for file_name in self.img_dir if is_image_file(file_name)]
        if logger is not None:
            logger.info(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        else:
            print(f'Finished creating an instance of {self.__class__.__name__} with {len(self.img_file_names)} examples')
        self.input_transform = input_transform

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self,i):
        if i % 30 == 0:
            imgu1 = np.array(Image.open(self.img_file_names[i]))
        else:
            imgu1 = np.array(Image.open(self.img_file_names[i - 1]))
        img = np.array(Image.open(self.img_file_names[i]))
        if i % 30 == 29:
            imgd1 = np.array(Image.open(self.img_file_names[i]))
        else:
            imgd1 = np.array(Image.open(self.img_file_names[i + 1]))

        pad = lambda img: np.stack([img,img,img],axis=2) if (len(img.shape)==2 or img.shape[2]==1) else img
        imgu1, img, imgd1 = pad(imgu1), pad(img), pad(imgd1)

        imgu1, img, imgd1 = self.input_transform(imgu1), self.input_transform(img), self.input_transform(imgd1)

        return imgu1, img, imgd1, self.img_file_names[i]


if __name__ == "__main__":
    from transform import ToLabel
    from torchvision.transforms import Compose, CenterCrop, Normalize
    from torchvision.transforms import ToTensor, ToPILImage
    input_transform = Compose([
        ToTensor(),
    ])
    target_transform = Compose([
        ToLabel(),
    ])
    dataset = SeqDataset('data/train.npy', 'data/masks', input_transform, target_transform)
    imgu1, img, imgd1, mask = dataset.__getitem__(0)
    print(imgu1.shape)
    print(img.shape)
    print(imgd1.shape)
    print(mask.shape)
