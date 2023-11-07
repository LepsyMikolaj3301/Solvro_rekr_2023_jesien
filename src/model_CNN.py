"""
This is the file containing the CNN Model

"than maybe your best course, would be to tread lightly..."
BB s5 e9

"""
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """

        This is the class containing the Model with the architecture responsible for learning

        ...

        Attributes
        ----------
        data : nd array
            a numpy array containing the data samples
        target : nd array
            a numpy array containing tha targets
        transform=None : a torch object
            an optional object which does: composure of transform methods

        Custom Methods
        -------
        forward(x: dataloader object):


    """
    def __init__(self):
        """
            The Model constructor
            Creates all necessary attributes for architecture

            !
            Thanks to the super() function it inherits EVERY method from nn.Module
            !

            ---
            :param conv1: the first Convolutional layer
            :param conv2: the second Convolutional layer
            :param conv2_drop: a dropout layer ( refer here: https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html)
            :param fc1: Fully connected layer 1
            :param fc2: Fully connected layer 2
            :param fc3: Fully connected layer 3
        """

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 48, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(48*4*4, 250)
        self.fc2 = nn.Linear(250, 75)
        self.fc3 = nn.Linear(75, 26)

    def forward(self, x):
        """
        THE ARCHITECTURE:

                            Channels, Height, Width     |   Channels_out
        -CONV_LAYER_1 ->       1       28      28       |       8
    After relu and maxpool:    8       24      24       |
        -CONV_LAYER_2 ->       8       24      24       |       8 * 6
    ----------------- ADDITIONALLY A DROPOUT LAYER HERE ----------------
    After relu and maxpool:    48      22      22       |

        FLATTENING THE CONVOLUTION:

        using nn.Linear()
        3 LINEAR LAYERS
        48 channels * 4 * 4 = 576
        576 neurons -> 250 neurons -> 75 neurons -> 26 neurons
                                        target neurons ^
        AFTER EVERY FULLY CONNECTED LAYER THERE ARE DROPOUT LAYERS ASWELL !!
        --------------------------------------------------------


        :param x: dataloader object:
        :return: a tensor shape: ( len(data), 1 )
                    containing one of the 26 targets as a prediction
                    softmax function: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 48*4*4)
        # flattening here ^
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        # returning a softmax function
        return F.softmax(x)
