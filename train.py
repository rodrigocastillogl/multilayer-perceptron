import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mlp import MLP
from dataset import train_dataset, train_data_loader

def train(model, dataloader, n_epochs, use_cuda = False):
    """
    Train model.
    Input
    -----
        * model: Module object.
        * dataloader: DataLoader object.
        * n_epochs : number of epochs during training.
        * use_cuda : flag to use cuda device.
    Output
    ------
    """

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD( model.parameters(), lr = model.lr )

    losses = []
    model.train()
    print(f'Training model for {n_epochs} epochs.')

    if use_cuda:
        print('Using cuda device...')

    for epoch in range(1, n_epochs+1):
        
        for batch, (X, y) in enumerate( dataloader ):
            
            if use_cuda:
                X = X.cuda()
                y = y.cuda()
            
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch} ----------------')
        print(f"\t Loss = {(loss):.4f}")
        losses.append(loss.item())

    return losses

if __name__ == '__main__':

    # Define model
    model = model = MLP( num_hiddens = 128,
                         num_outputs = 10 ,
                         lr = 5e-4)
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    
    # Train model
    epochs = 15
    losses = train(model, train_data_loader, epochs, use_cuda = use_cuda)

    # Save training image
    plt.figure( figsize = (6,4) )
    plt.plot( list(range(1, epochs+1)), losses, marker = 'o', c = 'forestgreen' )plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.tight_layout()
    plt.savefig('training.png')