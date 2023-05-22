import torch
from torch.utils.data import Dataset
from torchvision import datasets
from matplotlib import pyplor as plt


training_data = datasets.FashionMNIST(
    root = 'data'         ,
    train = True          ,
    download = True       ,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = 'data'         ,
    train = False         ,
    download = True       ,
    transform = ToTensor()
)

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
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        print(sample_idx)
        img, label = training_data[sample_idx]
        figure.add_subplot(3, 3, i+1)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow( img , cmap = 'gray' )
    plt.tight_layout()
    #plt.savefig('imgs/examples.png')