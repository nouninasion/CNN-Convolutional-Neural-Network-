Key Components of CNN
CNNs are made up of several types of layers that work together to extract features and make predictions.

1. Convolutional Layer
This is the core component of a CNN. This layer performs a "convolution" operation on the input.

Filter (Kernel): A small matrix (e.g., 3x3 or 5x5) that slides across the input image. Each filter is designed to detect specific features such as edges, textures, or shapes.

Convolution Operation: The filter is multiplied element-wise with the part of the image it covers, and the results are summed to produce a single value in the output feature map. This process repeats across the entire image.

Feature Map: The output of the convolution operation. Each filter produces a separate feature map that shows where the detected feature appears in the input image.

Padding: Adds zero-value pixels around the image edges to control the output size.

Stride: The number of steps the filter moves at each slide. Larger strides result in smaller output feature maps.

2. Activation Function
After convolution, a non-linear activation function is applied to the feature maps.

ReLU (Rectified Linear Unit): This function converts all negative values to zero while leaving positive values unchanged:
f(x) = max(0, x).
This introduces non-linearity, allowing the network to learn complex patterns.

3. Pooling Layer
This layer reduces the spatial dimensions (width and height) of the feature maps, decreasing the number of parameters and computation, and helping to prevent overfitting.

Max Pooling: The most common type; it selects the maximum value from each small window (e.g., 2x2) in the feature map.

Average Pooling: Takes the average value instead of the maximum.

4. Fully Connected Layer (Dense Layer)
After several convolutional and pooling layers, the extracted feature maps are flattened into a single long vector.

This vector is fed into one or more fully connected layers, similar to traditional neural networks.

These layers are responsible for performing classification or regression based on the high-level features extracted by previous layers.

The final output layer usually uses Softmax (for multi-class classification) or Sigmoid (for binary classification) activation functions.

How CNN Works (Workflow)
Let’s say you want to train a CNN to recognize whether an image contains a cat or a dog:

Input Image: An image (e.g., 224x224 pixels with 3 RGB channels) is fed into the CNN.

Feature Extraction (Convolution & Pooling Layers):

First Convolutional Layer: Small filters (e.g., 3x3) slide over the image to detect very basic features like horizontal, vertical, or diagonal edges. The output is a set of feature maps.

Activation Function (ReLU): Non-linearity is applied to the feature maps.

First Pooling Layer: Reduces the size of the feature maps (e.g., from 224x224 to 112x112) while retaining the most important features.

This process (Convolution → Activation → Pooling) is repeated several times. Each deeper convolutional layer learns more complex features by combining patterns detected by earlier layers—such as detecting eyes, noses, or ears.

The deeper the layer, the more abstract and complex the extracted features become.

Flattening: After passing through several convolution and pooling layers, the final feature maps are flattened into a single long vector.

Classification (Fully Connected Layers):

This feature vector is fed into one or more fully connected layers. These layers learn to combine the extracted features and make the final decision.

Output Layer: The final layer (e.g., with 2 neurons for "cat" and "dog" and softmax activation) outputs the probability of the image being a cat or a dog.

Training: During training, the network is shown many labeled examples of cats and dogs. Using algorithms like backpropagation and gradient descent, the weights of the filters and neurons are adjusted to minimize prediction errors.

