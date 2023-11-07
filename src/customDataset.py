import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


def one_hot_2_value(target_array):
    """

    this function translates the one-hot encoded targets into a numpy array of values of this target
    we use the np.nonzero() method to extract the index with the "one" in it,
    then we transpose it to fit the data

    :param target_array: a numpy array of one-hot encoded values ( an array of 0's and a 1)
    :type target_array: nd array
    :return: target_array: a transposed numpy array of values
    :rtype target_array: nd array
    """
    return np.transpose(np.nonzero(target_array)[1])


class UppercaseAlphabetDataset(Dataset):
    """

    A Custom Dataset class inheriting from the torch.utils.data.Dataset

    ...

    Attributes
    ----------
    data : nd array
        a numpy array containing the data samples
    target : nd array
        a numpy array containing tha targets
    transform=None : a torch object
        an optional object which does: composure of transform methods

    Methods
    -------
    __len__():
        returns the length of the dataset.

    __getshape__():
        returns the shape of the dataset

    __getitem__(idx: int):
        returns a pair of values consisting of: ( data_sample, the_target )

    """
    def __init__(self, data, target, transform=None):
        """

        The datasets constructor
        Creates all necessary attributes for the dataset object

        :param data: the samples
        :param target: the targets
        :param transform: Composed transforms
        """
        self.data = data
        self.target = torch.from_numpy(one_hot_2_value(target))
        self.transform = transform
        self.shape = self.__getshape__()

    def __len__(self):
        return len(self.target)

    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape)

    def __getitem__(self, idx: int):
        """
        this method is used to access data and targets in the dataset
        IT OVERRIDES THE METHOD __getitem__ IN THE Dataset CLASS ! ! !

        ! ! !
        after applying transforms:
            x = prepare image for data augmentation a.k.a. transpose it and change it to a PIL
            apply transforms
        ! ! !

        :param idx: the index of the dataset
        :return: a pair of variables: ( data_sample, target )
        """
        x = self.data[idx]
        y = self.target[idx]

        if self.transform:
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C x H x W
            x = Image.fromarray(self.data[idx].astype(np.uint8).transpose(0, -1))
            x = self.transform(x)

        return x, y
