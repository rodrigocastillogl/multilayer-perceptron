# Multilayer Perceptron

> by: Rodrigo Castillo González

## Multilayer Perceptron

A *Multilayer Perceptron* (MLP) is one of the simplest deep models. It consists of:
* An input layer: features of the input variable.
* Hidden layers: that compute over the input features; it is important to introduce non-linearlity with activation functions in these layers,
this way the model works as "universal function approximator" .
* Output layer: output of the model.

The following figure shows the model (from: [(Neurons in Neural Networks, by Nikhil Bhargav)](https://www.baeldung.com/cs/neural-networks-neurons).

<img src="/imgs/mlp.png" alt="Multi-layer Perceptron" width="70%" height="70%">

We can refer to this kind of architecture as *fully conected layers* because the output of every neuron in a layer is conected to every neuron in the following layer.

# Forward Propagation

*Forward propagation* or *forward pass* refers to the computation and storage of intermediate variables for a neural network in order from the input layer to the output layer.

Let $\bf{x}$ de the input variable, ${\bf{W}}^{(i)}$ and ${\bf{b}}^{(i)}$ the weights and bias of the $i$-th layer respectively (starting counting from the first hidden layer), 
$\sigma^{(i)}$ the activation function of the $i$-th layer, ${\bf{h}}^{(i)}$ the output of the $i$-th hidden layer and $\bf{o}$ the output of the model. Then, forward propagation
is computed as follows:

$$\begin{matrix}
{\bf{h}}^{(1)} & = &  \sigma^{(1)} ( \ {\bf{W}}^{(1)} \bf{x} + {\bf{b}}^{(1)} \ ) \\
{\bf{h}}^{(2)} & = & \sigma^{(2)} ( \ {\bf{W}}^{(2)} {\bf{h}}^{(1)} + {\bf{b}}^{(2)} \ ) \\
\vdots & = & \vdots \\
{\bf{h}}^{(L-1)} & = & \sigma^{(L-1)} ( \ {\bf{W}}^{(L-1)} {\bf{h}}^{(L-2)} + {\bf{b}}^{(L-1)} \ ) \\
\bf{o} & = & {\bf{W}}^{(L)} {\bf{h}}^{(L-1)} + {\bf{b}}^{(L)} \\
\end{matrix}$$

## Fashion-MNISt classification

Describe example

## References
* [Dive into Deep Learning](https://d2l.ai/)
* [Pytorch Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
