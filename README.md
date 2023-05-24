# Multilayer Perceptron

> by: Rodrigo Castillo González

## Multilayer Perceptron

A Multilayer Perceptron (MLP) is one of the simplest deep models. It consists of:
* An input layer: features of the input variable.
* Hidden layers: that compute over the input features; it is important to introduce non-linearlity in these layers, this way the model works as "universal function approximator" .
* Output layer: output of the model.

The following figure shows the model (from: [(Neurons in Neural Networks, by Nikhil Bhargav)]https://www.baeldung.com/cs/neural-networks-neurons() ).

<img src="/imgs/mlp.png" alt="Multi-layer Perceptron" width="50%" height="50%">

# Forward Propagation

Forward propagation or forward pass refers to the computation and storage of intermediate variables for a neural network in order from the input layer to the output layer.

Let $\bf{x}$ de the input variable, $\bf{W}^{(i)}$ the weights of the $i$-th hidden layer, $\bf{h}$ the output of the $i$-th hidden layer and $\bf{o}$ the output of the model. Then, forward propagation is computed as follows:

* 

## Fashion-MNISt classification

Describe example

## References
* [Dive into Deep Learning](https://d2l.ai/)
* [Pytorch Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
