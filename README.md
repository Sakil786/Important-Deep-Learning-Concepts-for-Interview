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

# What is an Activation Function?
* In a neural network, after the multiplication of features by weights and the addition of bias, an activation function is applied.
* Activation functions transform the output of this initial operation.
* They are important because they determine whether a neuron gets activated or not.
* If a neuron is activated, it means it is transferring a signal that helps in the classification of the final output.
* If a neuron is not activated, no signal is transferred.
* The array can be seen as a kind of activation function.
## Sigmoid Activation Function
* The Sigmoid activation function is a type of activation function.
* It transforms the input value (Y, which is the product of weights and input features plus bias) to a value between 0 and 1.
* The formula for the Sigmoid function is 1 / (1 + e^(-Y)).
* If the output of the Sigmoid function is less than 0.5, it is generally considered as 0 (neuron not activated).
* If the output is greater than 0.5 (or equal to, though not explicitly mentioned), it is considered as 1 (neuron activated).
* The Sigmoid function is often used in the final output layer for classification problems because it transforms the value into a probability-like range between 0 and 1.
  
## ReLU (Rectified Linear Unit) Activation Function
* ReLU is another activation function.
* The formula for ReLU is max(Y, 0), where Y is the output of the weights and features plus bias.
* If Y is negative, the output of the ReLU function is 0 (neuron not activated).
* If Y is positive, the output of the ReLU function is the positive value of Y.
* ReLU is described as being more popular than Sigmoid.
* It is often used in the middle layers of neural networks, and sometimes in all layers except the output layer for regression problems.
* However, for classification problems, Sigmoid is generally preferred in the final output layer
