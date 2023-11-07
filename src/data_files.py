"""
This file is used in main.py for:
- Data augmentation
- Class weighing
- setting up the data in datasets
- inserting the data in DataLoaders ( which are batching and assigning workers )
"""

import torch.utils.data as data_utils
from torchvision import transforms
import numpy as np
from collections import Counter
from customDataset import UppercaseAlphabetDataset, one_hot_2_value
import zipfile
import os


def unziping():
    """

    a void that uses the zipfile to extract files from the ZIP-file

    when a file is already unpacked, skips it in the for loop
    """

    the_dir = r'data_folder'
    the_zip = r'solvro-rekrutacja-zimowa-ml-2023.zip'
    with zipfile.ZipFile(the_dir + r'/' + the_zip, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # if ! a file exists or ! there is a file named like that
            if not os.path.exists(the_dir + r'/' + member) or not os.path.isfile(the_dir + r'/' + member):
                zip_ref.extract(member, the_dir)
                print(f"Extracted: {member}")


def calculation_of_weights(label_array) -> list:
    """

    This function calculates the wights of the target numpy array ( used in training the model )

    :param: label_array
    :type: ndarray
    :return: target_weights
    :rtype: list
    """
    target_weights = []
    # Counting how many targets appear in the target set
    the_counter = Counter(one_hot_2_value(label_array))
    for _, occurrence_count in sorted(the_counter.items()):
        # when they are no values of an occurance, we should add a neutral weight to it
        if occurrence_count < 0:
            target_weights.append(0.5)
        else:
            # we calculate the weight as the inverse of its occurances
            target_weights.append(round(1. / occurrence_count, 5))
    return target_weights


def data_sampler(dataset, target_weight_array: list) -> list:
    """

    This function ASSIGNS target weights to every sample ( every data image )

    :param dataset: The ( training ) dataset
    :type dataset: torch dataset
    :param target_weight_array: A list of weights for every target
    :type target_weight_array: list
    :return sample_weights: A list with len of the dataset, with the target weights assigned
    :rtype: list
    """
    # here we create a list of zeroes with the len of the training data
    sample_weights = [0] * len(dataset)

    for i, (data, target) in enumerate(dataset):
        target_weight = target_weight_array[target]
        sample_weights[i] = target_weight

    return sample_weights

def ndarray_2_tensor(array):
    """

    this function is for exceptions in the data like x_test, where the program has to predict the values
    this shouldn't be used in datasets, as they have their own transform as a Custom Dataset Class
    refer to file customDataset.py

    :param array: a DATA !!! numpy array
    :type array: ndarray
    :return same array: The same array changed to a torch Tensor ( dtype = float32 )
    :rtype torch
    """

    trans_to_tensor = transforms.Compose([transforms.ToTensor()])
    return trans_to_tensor(np.moveaxis(array, 0, -1))


def datasets() -> dict:
    """

    this function:
    - reads data from the 'solvro-rekrutacja-zimowa-ml-2023.zip' file
    - creates variables as numpy arrays
    - contains the my_transform_train object used for DATA AUGMENTATION
    - arranges the data and targets as TRAIN, VALIDATION datasets
                                                AND
                                        TEST_DATA tensor

    *** THIS FUNCTION USES A CUSTOM DATASET ***
    refer to customDataset.py

    :return dataset_dict: a dictionary containing the datasets and the data tensor
    :rtype dataset_dict: dict
    """


    # the training data
    X_train = np.load(r'data_folder/X_train.npy')
    Y_train = np.load(r'data_folder/y_train.npy')

    # validation data
    X_val = np.load(r'data_folder/X_val.npy')
    Y_val = np.load(r'data_folder/y_val.npy')

    # transform object
    my_transform_train = transforms.Compose([
        # data augmentation didn't increase the Loss score, as a sideways "I" isn't the same letter lol

        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=45),

        # but cropping worked with quite good results
        transforms.RandomResizedCrop(28, scale=(0.8, 0.9)),
        transforms.ToTensor(),

        # transforms.Normalize(mean=0.485, std=0.229)
        # Normalising could work, but it didn't
        ])

    dataset_dict = {
        # creating the training dataset
        'training_dataset': UppercaseAlphabetDataset(X_train,
                                                     Y_train,
                                                     transform=my_transform_train),
        # creating the validation dataset
        'validation_dataset': UppercaseAlphabetDataset(X_val,
                                                       Y_val,
                                                       transform=transforms.ToTensor()),
        # creating the test_data tensor for future prediction test, refer to main for more intel
        'test_data': ndarray_2_tensor(np.load(r'data_folder/X_test.npy'))

    }

    return dataset_dict


def loader() -> dict:
    """

    this function returns a dictionary with loaders
    Then those loaders are used for "feeding" the data to the CNN Model

    the function creates:
     - a variable dt_set containing a DICTIONARY of datasets
     - a data sampler for class weighing ( using WeightedRandomSampler ) !!!

    the training dataloader is sampled!
    the validation dataloder isn't !
    :return loaders: A dictionary containing different loaders
    :rtype loaders: dict
    """

    dt_set = datasets()

    class_weights = calculation_of_weights(np.load(r'data_folder/y_train.npy'))

    sample_of_data_weight = data_sampler(dt_set['training_dataset'], class_weights)

    sampler = data_utils.sampler.WeightedRandomSampler(sample_of_data_weight,
                                                       num_samples=len(sample_of_data_weight),
                                                       replacement=True
                                                       )

    loaders = {

        # if the training takes too long -> decrease the amount of loaders and the batch size
        'train': data_utils.DataLoader(dt_set['training_dataset'],
                                       batch_size=200,
                                       shuffle=False,
                                       num_workers=2,
                                       sampler=sampler
                                       ),

        'valid': data_utils.DataLoader(dt_set['validation_dataset'],
                                       batch_size=200,
                                       shuffle=True,
                                       num_workers=2)

    }
    return loaders
