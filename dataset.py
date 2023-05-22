import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import pandas as pd
import os
from matplotlib import pyplot as plt

def download_fashion_mnist(data_path):
    """
    Download Fashion MSNIT data from PyTorch.
    Input
    -----
        data_path: data directory.
    Output
    ------
        None.
    """

    datasets.FashionMNIST( root  = data_path ,
                           train = True      ,
                           download = True   )
    datasets.FashionMNIST( root  = data_path ,
                           train = False     ,
                           download = True   )

class Dataset_mnist(Dataset):
    """
    MNIST Dataset class.
    """

    def __init__(self, features_path, labels_path):
        """
        Constructor
        Input
        -----
            * features_path: features file path.
            * labels_ path: labels file path.
        Output
        ------
            None
        """

        # X = 
        pass


labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

if __name__ == '__main__':

    data_path = 'data'

    if os.path.exists(data_path):
        print("Data directory already exists.")
    else:

        # Download data
        download_fashion_mnist(data_path)
        


    """
    # plot some examples
    figure = plt.figure( figsize = (6, 6) )
    for i in range(9):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        print(sample_idx)
        img, label = training_data[sample_idx]
        figure.add_subplot(3, 3, i+1)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow( img.squeeze() , cmap = 'gray' )
    plt.tight_layout()
    plt.savefig('imgs/examples.png')
    """
