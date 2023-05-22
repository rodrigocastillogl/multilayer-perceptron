import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import os
from matplotlib import pyplot as plt

train_dataset = datasets.FashionMNIST( root  = 'data'  ,
                                       train = True    ,
                                       download = True ,
                                       transform = ToTensor() )
test_dataset = datasets.FashionMNIST( root  = 'data'  ,
                                      train = False   ,
                                      download = True ,
                                      transform = ToTensor() )

train_data_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_data_loader  = DataLoader(test_dataset, batch_size = 64, shuffle = True)

    
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

    # plot some examples
    figure = plt.figure( figsize = (6, 6) )
    for i in range(9):
        sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
        print(sample_idx)
        img, label = train_dataset[sample_idx]
        figure.add_subplot(3, 3, i+1)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow( img.squeeze() , cmap = 'gray' )
    plt.tight_layout()
    plt.savefig('imgs/examples.png')

    # Display batch sizes
    train_features, train_labels = next(iter(train_data_loader))
    print( f'Features batch shape: {train_features.size()}' )
    print( f'Features batch shape: {train_labels.size()}' )