# Important-Deep-Learning-Concepts-for-Interview
# How a neural network works:
* Input features (like x1, x2, x3) are fed into the neural network.
* Weights (w1, w2, w3) are assigned to the connections between the input features and the neurons. These weights play a very important role.
* Inside each neuron, a two-step operation occurs:
* Summation of the products of the inputs and their corresponding weights (e.g., Y = W1X1 + W2X2 + W3X3).
* A bias term (a smaller value) is added to this summation.
* The result of this calculation is then passed through an activation function.
* The activation function (e.g., sigmoid) transforms the input into an output, often between a specific range (e.g., 0 to 1 for sigmoid).
* If the output of the activation function exceeds a certain threshold (e.g., 0.5 for sigmoid), the neuron is considered activated. This activation determines if the signal is passed on.
* This process of weighted summation and activation is applied in each and every neuron of the network.
* The output from one layer of neurons becomes the input for the next layer, with its own set of weights.
* Finally, the signal reaches the output layer, where another activation function is applied to produce the final result (e.g., a classification of 0 or 1).
* This entire process of information flowing from the input to the output is called forward propagation.
* The weights determine which neurons are triggered.
* During training, the weights are updated using a process called backpropagation
