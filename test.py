import torch
import torch.nn as nn
from mlp import MLP
from dataset import test_dataset, test_data_loader

def test(model, dataloader, use_cuda):
    """
    Test model
    Input
    -----
        * model : 
        * dataloader :
        * use_cuda :
    Output
    ------
        * acc: model accuracy.
    """

    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    acc  = 0
    loss = 0

    with torch.no_grad():
        for X, y in dataloader:

            if use_cuda:
                X = X.cuda()
                y = y.cuda()
            
            pred = model(X)
            acc += (pred.argmax(1) == y).sum().item()
            loss += loss_fn(pred, y).item()
            
    acc /= len(dataloader.dataset)
    loss /= len(dataloader)

    return acc, loss

if __name__ == '__main__':

    # Define model
    model = MLP( num_hiddens = 128 ,
                 num_outputs = 10  ,
                 lr = 5e-4         )
    # load weights
    weights_path = "weights.pth"
    model.load_state_dict(torch.load(weights_path))

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    
    # Test model
    test_acc, test_loss = test(model, test_data_loader, use_cuda)
    print(f"Test Error: \n Accuracy: {(100*test_acc):0.2f}%, Average loss: {test_loss:0.5f}")