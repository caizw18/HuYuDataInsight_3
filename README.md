# HuYuDataInsight_3
This is the workload for our company HuYuDataInsight LLC from Apr 1, 2024 to Apr 12, 2024

We write the Python code to perform CIFAN-10 images classification in PyTorch by ResNet based on standard block. We train the model by GPU. 

In the Python code, we define the class StandardBlock. In this class, we firstly define a variable “expansion”, which represents how many times the 
output dimension in the residual structure is the input dimension. For StandardBlock, it doesn't increase the dimension, so expansion is equal to 1.

Then we define an instance method to initialize a newly created StandardBlock object. We define conv1 as the first 2D convolution layer with kernel 
size 3 and padding 1. Then bn1 is the first batch normalization. Then conv2 is the second 2D convolution layer with kernel size 3 and padding 1. Then
bn2 is the second batch normalization. We also define a shortcut to make the addition (+) successful.

In the forward function, the input x will at first go through conv1, then bn1, then ReLU, then conv2, then bn2. The residual structure goes through
the ReLU layer after addition.
