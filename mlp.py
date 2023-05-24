import torch
import torch.nn as nn

# Define model
class MLP(nn.Module):
    
    """
    MultiLayer perceptron for Fashion MNIST classification.
    """

    def __init__(self, num_hiddens, num_outputs, lr = 1e-3):
        """
        Constructor
        Input
        -----
            * num_hidden: neurons in the hidden layer.
            * num_outputs: number of outputs.
            * lr : learning rate.
        Output
        ------
            None
        """

        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        self.lr = lr

        # define model
        self.flatten = nn.Flatten()
        self.hidden = nn.Sequential( nn.Linear(28*28, self.num_hiddens) ,
                                     nn.ReLU()                          )
        self.output = nn.Linear(self.num_hiddens, self.num_outputs)
    
    def forward(self, x):
        """
        Forward pass.
        Input
        -----
            * x: input 28x28 image
        Output
        ------
            * Model evaluation
        """

        x = self.flatten(x)
        x = self.hidden(x)
        logits = self.output(x)

        return logits
    
if __name__ =='__main__':

    # Define model
    model = MLP( num_hiddens = 128,
                 num_outputs = 10 ,
                 lr = 5e-4)
    
    print(model)