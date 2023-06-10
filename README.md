# Multilayer Perceptron

## Multilayer Perceptron

A *Multilayer Perceptron* (MLP) is one of the simplest deep models. It consists of:
* An input layer: features of the input variable.
* Hidden layers: that compute over the input features; it is important to introduce non-linearlity with activation functions in these layers,
this way the model works as "universal function approximator" .
* Output layer: output of the model.

The following Figure (from: [Neurons in Neural Networks, by Nikhil Bhargav](https://www.baeldung.com/cs/neural-networks-neurons)) shows the model.

<img src="/imgs/mlp.png" alt="Multi-layer Perceptron" width="70%" height="70%">

We can refer to this kind of architecture as *fully connected layers* because the output of every neuron in a layer is connected to every neuron in the following layer.

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


## Activation Functions

Some activation functions are:

* *Rectified Linear function* (ReLU):

Defined as
$$\mathrm{ReLU}(x) = \max{(0, x)}$$

It retains only positive values, and discard all negavite values their activations to $0$. For positive arguments the derivative of the ReLU
function is $1$ and for negative arguments it is $0$. Although it is not differentiable when $x=0$, by default we use the left-hand-side derivative
and say that the derivative is $0$ when $x=0$.

There exists some variations of this function: *Leaky ReLU*, *parameterized ReLU* (pReLU) and some others.

* *Sigmoid function*:

Defined as
$$\mathrm{sigmoid}(x) = \frac{ 1 }{ 1+\mathrm{exp}(-x) }$$

It squashes any input in $\mathbb{R}$ to some value in the range $[0,1]$ Its derivative can be expressed as

$$\frac{\mathrm{d}}{\mathrm{d}x} \ \mathrm{sigmoid}(x) = \mathrm{sigmoid}(x) \left( 1 - \mathrm{sigmoid}(x) \right)$$

When the input diverges from $0$ in either direction, the derivative approaches $0$, and this can lead to a vanishing gradient problem.

* *Tanh function*:

It is the *hyperbolic tangent function*, and it is defined as

$$\mathrm{tanh}(x) = \frac{ 1 - \mathrm{exp}(-2x) }{ 1 + \mathrm{exp}(-2x) }$$

Similar to sigmoid function, it squashes any input in $\mathbb{R}$ to some value in the range $[-1,1]$.

Its derivative can be expressed as

$$\frac{ \mathrm{d}}{\mathrm{d}x} \ \mathrm{tanh}(x) = 1 - {\mathrm{tanh}}^2(x)$$

Also, when the input diverges from $0$ in either direction, the derivative of the Tanh function approaches $0$.

## Back Propagation

*Back propagation*

*Automatic differentiation* and *computational graphs* profoundly simplifies the implementation of deep model.

## Fashion-MNISt classification

Describe example

## References
* [Dive into Deep Learning](https://d2l.ai/)
* [Pytorch Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
