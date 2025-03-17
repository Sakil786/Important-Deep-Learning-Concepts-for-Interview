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
## Leaky Relu:
* Leaky Relu is a modification of the Relu activation function.
* It aims to address the "dying Relu" problem. This is a situation where neurons stop updating their weights because their output remains zero.
* For negative input values, instead of outputting zero like the standard Relu, Leaky Relu outputs a small non-zero value (alpha * x).
* This small non-zero output for negative inputs ensures that there is still a small gradient, preventing the neuron from becoming completely inactive.
* The key difference from Relu is the behaviour for negative inputs, where Leaky Relu allows a small, linear output.
 ## GELU activation function:
* **Mathematical Representation**: The GELU activation function is mathematically represented as the product of the input $x$ and the cumulative distribution function (CDF) of the standard Gaussian distribution, denoted as $\Phi(x)$. So, $GELU(x) = x \cdot \Phi(x)$.
* **Approximation in GPT-2**: Due to the complexity of computing the Gaussian CDF, an approximation was used for training GPT-2. This approximation is: $GELU(x) \approx 0.5 \cdot x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$.
* **Smoothness**: Unlike ReLU, the GELU activation function is smooth throughout. ReLU has a sharp corner at $x=0$, making it non-differentiable at that point. GELU, being smooth, is differentiable across all $x$.
* **Non-Zero for Negative Inputs**: For $x < 0$, the ReLU activation function returns zero. In contrast, GELU produces small non-zero output values for negative inputs. It tends towards zero but does not become zero.
* **Prevents Dead Neuron Problem**: The fact that GELU is not zero for negative $x$ helps to prevent the dead neuron problem associated with ReLU. Neurons receiving negative input can still contribute to the learning process with GELU, unlike ReLU where they would output zero and effectively become inactive.
* **Performance in LLMs**: Experiments have generally shown that GELU performs better than ReLU in the context of large language models (LLMs). The smoothness of GELU can lead to better optimisation properties during training, allowing for more nuanced adjustments to model parameters.
* **Incorporated in Feed Forward Network**: In the Transformer block of LLM architectures like GPT, the GELU activation function is typically found within the feed forward neural network. This network consists of two linear layers with the GELU activation applied after the first linear layer

# Vanishing Gradients:
* A common problem during neural network training, affecting both regular Artificial Neural Networks (ANNs) and Recurrent Neural Networks (RNNs).
* During backward propagation, weights in the network are updated based on the gradient of the loss function with respect to those weights.
* The gradient for earlier layers is calculated using the chain rule, involving the multiplication of derivatives across multiple layers.
* If these individual derivatives are small, their product becomes even smaller.
* A very small gradient leads to a minimal change in the weights of the earlier layers during training.
* This results in a slow learning process because the earlier layers hardly learn anything.
* In deep neural networks, the problem is more prominent due to a greater number of layers.
* In RNNs, vanishing gradients cause a short memory, as the impact of earlier inputs in a sequence diminishes over time, making it difficult to learn long-range dependencies.
# Exploding Gradients:
* Occurs when the individual derivatives during backpropagation are large.
* The product of these large derivatives results in a very large gradient.
* Mentioned as a problem in RNNs.

# Optimizers

## Gradient Descent (GD):
* Updates weights by considering all data points in the dataset to calculate the derivative of the loss function.
* The weight update formula is: W_new = W_old - learning rate * (derivative of loss with respect to W_old).
* Computationally expensive and requires significant resources for large datasets as all records need to be loaded.
* The convergence path towards the global minima is typically more direct.
## Stochastic Gradient Descent (SGD):
* Updates weights by considering only one data point at a time.
* The loss calculation for weight update considers a single data point: Loss = (Y - Y_hat)^2.
* Used in scenarios like linear regression.
## Mini-batch Stochastic Gradient Descent (Mini-batch SGD):
* Updates weights by considering a small batch (k) of data points, where k is less than the total number of data points (n).
* The loss is calculated based on the batch of k data points: Loss = Summation from i=1 to k of (Y_i - Y_hat_i)^2.
* Commonly used in many neural network techniques like CNNs.
* Less computationally expensive than GD as it processes smaller batches of data.
* The convergence path towards the global minima is more zigzag or noisy compared to GD due to the sample-based updates.
* The derivative of the loss with respect to the weights using Mini-batch SGD is an approximation of the derivative calculated using GD (which considers the entire population of data).
* Choosing the appropriate batch size (k) depends on the computational power available.
* Mini-batch SGD can be likened to taking a sample of the population (GD) to estimate the mean/average.
* The "zigzag movement" in convergence is considered noise, which can be addressed by techniques like Stochastic Gradient Descent with Momentum .
## Stochastic Gradient Descent with Momentum
* Momentum in SGD is a technique used to smoothen the zigzag path of weight optimisation.
* This smoothing is achieved by incorporating a velocity term (VdW) into the weight update rule.
* The weight update rule with momentum is: W_new = W_old - η * VdW.
* The velocity term (VdW) is calculated as an exponentially weighted average of past gradients (dW).
* The formula for VdW is: VdW = β * VdW_prev + (1 - β) * dW.
* This means that recent gradients have a higher weight in determining the direction of the next step, while older gradients have a smaller influence.
* The parameter beta (β) is called the momentum coefficient or momentum term.
* It controls the weight given to previous velocity values and the current gradient.
* The source suggests that an optimal value for beta is generally 0.9.
* By using momentum, the optimisation process becomes less erratic and can potentially outperform standard SGD.
* The underlying concept behind momentum is similar to exponential smoothing used in time series analysis, where recent observations are given more weight.
## Ada Grad:
* Ada Grad uses different learning rates for different weights (parameters) in a neural network.
* It adapts the learning rate based on the history of gradients for each weight.
* Weights with frequently large gradients will have their learning rate decreased over time.
* Weights with infrequent or small gradients will have their learning rate increased over time.
* The update rule for a weight involves dividing the initial learning rate by the square root of the cumulative sum of squared past gradients for that weight.
* This helps in situations with sparse data, where some features are updated more than others.
* Initially, it takes larger steps towards the minimum, and as it gets closer, it takes smaller steps.
* A potential issue with Ada Grad is that the cumulative sum of squared gradients can become very large, causing the learning rate to become very small, potentially halting learning.
## Ada Delta:
* Ada Delta addresses the issue of the learning rate becoming too small in Ada Grad.
* Instead of accumulating all past squared gradients, Ada Delta uses an exponentially weighted average of past squared gradients.
* This means that more recent gradients have a higher influence on the current learning rate adaptation.
* Ada Delta maintains a moving average of the squared gradients.
* This helps to prevent the learning rate from decaying too aggressively.
* The update involves a decay factor (often denoted as rho) to control the window of past gradients considered.

