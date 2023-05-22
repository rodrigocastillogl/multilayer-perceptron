import torch
import os
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import wget
import zipfile

def load_data(data_url, data_path):
    """
    Download data set.
    Input
    -----
        * data_url  : url to download data.
        * data_path : data directory path.
    Output
    -----
        * None
    """

    print(f'Downloading data from: {data_url}')
    filename= wget.download(data_url, data_path)
    print(f'Saved as {data_path + filename }')


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

    data_url = 'https://www.kaggle.com/datasets/zalando-research/fashionmnist/download?datasetVersionNumber=4'
    data_path = 'data'

    if os.path.exists(data_path):
        print('Data already exists.')
    
    else:
        
        # Create data directory
        os.makedirs(data_path)
        
        # Download data
        load_data(data_url, data_path)


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
