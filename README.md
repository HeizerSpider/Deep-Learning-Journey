# Deep-Learning-Journey

1) [Deep Learning Theory (Continuous update)](https://github.com/HeizerSpider/Deep-Learning-Journey/blob/master/README.md#deep-learning-theory)
2) [Conda environment setup for MacOS (Final)](https://github.com/HeizerSpider/Deep-Learning-Journey/blob/master/README.md#conda-environment-setup-for-macos)

## Deep Learning Theory

a) [Overview of Deep Learning Theory](https://github.com/HeizerSpider/Deep-Learning-Journey/blob/master/README.md#a-overview-of-deep-learning-theory)  
b) [Computer Vision Pipeline](https://github.com/HeizerSpider/Deep-Learning-Journey/blob/master/README.md#b-computer-vision-pipeline)  
c) [Deep Learning and Neural Networks](https://github.com/HeizerSpider/Deep-Learning-Journey/blob/master/README.md#c-deep-learning-and-neural-networks)  

### a) Overview of Deep Learning Theory
Artificial neural networks are a biomimicry of the human brain and its neurons, 
with both having input signals, a flow of information and outputs. Might be good to look to 
the human brain to improve the way the Aritificial Neural Networks (ANNs) are made.  

|Human|AI|
|-----|--|
|Eyes|Camera|
|Neurons|Nodes|
|Dendrites|Input|
|Synapse|Output|

These sensors are not limited to cameras only, but also include [lidar](https://medium.com/@SmartLabAI/3d-object-detection-from-lidar-data-with-deep-learning-95f6d400399a) 
and radar sensors (3d object detection) which can used as inputs to the model.
In the case of Autonomous Vehicles, depending on the company (mostly), a different 
variety of sensors are used for the purpose of mapping and localisation.
Some of these sensors include IMUs, GPS and ultrasonic sensors.
However for the purpose of deep learning and computer vision, the main focus will be on cameras/image input.


### b) Computer Vision Pipeline

i) [Image input](https://github.com/HeizerSpider/Deep-Learning-Journey/blob/master/README.md#i-image-input)  
ii) [Pre-Processing](https://github.com/HeizerSpider/Deep-Learning-Journey/blob/master/README.md#ii-pre-processing)  
iii) [Feature extraction](https://github.com/HeizerSpider/Deep-Learning-Journey/blob/master/README.md#iiiiv-feature-extraction-and-classifier)  
iv) [Classifier/ML Model](https://github.com/HeizerSpider/Deep-Learning-Journey/blob/master/README.md#iiiiv-feature-extraction-and-classifier)

----------------------------------------------------------------------------------------------------------------

#### i) Image input  
Images can be interpreted as a function of two variables, x and y (2 dimensional area), 
ie image is divided into grids (pixels), and each of these pixels 
are assigned a number between 0 to 255 that represents the intensity/brightness of that pixel.

Colour images have 3 values (all between 0-255)-Red, Green, Blue (3 dimensions-(x, y, 3))  
Colour is not always important so do be mindful of the purpose of the model and 
use/prepare the datasets to minimize the amount of time to train the model (reducing complexity)

----------------------------------------------------------------------------------------------------------------

#### ii) Pre-processing  
Essentially preparing the dataset before training/testing the model  
- Standardizing the images . 
- Transforming the colors: Reduce complexity (mentioned above)  
- Data augmentation: Augmenting datasets with different version of the intial images 
(scaling/rotatingflipping, de-texture/colourize etc.) meant to enlarge dataset and 
exposes neural networks to a greater variety of images and hence able to recognize 
objects in multiple forms.  
- Other Techniques: Can be dependent on the need of the project (removing background, 
increasing/decreasing noise etc.)  

----------------------------------------------------------------------------------------------------------------

#### iii/iv) Feature extraction and Classifier  
CORE COMPONENT  
Purpose:   
- extract useful features for defining/distinguishing an object in an image properly  
(eg. car: useful features include wheels, door, spoiler etc.)  
- transform the data into features vector-list of features (for learning algorithm 
to learn the characteristics of the object)  
- features must be useful (defining features that allow for the return of a higher probability to that 
given class through the use of such a feature-quite intuitive, just like how we humans 
would distinguish between different objects)  
- How are features extracted? (Classification Task)

----------------------------------------------------------------------------------------------------------------

- Old ML method (manual hand crafted feature set: HOG, HAAR cascades, SIFT, SURF)  
Input -> Feature Extraction (using abovementioned feature sets) -> Learning Algorithm(SVM/Adaboost) -> Output  

- Deep learning on the other hand automatically extracts features (how???)
The network extracts features (all) and learns their importance on the output by applying 
weights to its connections (removes unnecessary info)  
Feed raw image -> Image passes through network layers and identifies the patterns within 
the image to create features -> Output  
Hence, neural networks are essentially feature extractors + classifiers that are end-to-end trainable  
Usefulness of features depends on the weights assigned to them (Higher weights==greater impact on outputs)  
Neural networks automatically extract useful features when the input image passes through the layers of 
the neural network to learn their features layer-by-layer. The deeper the layer is (more layers), the more it 
will learn the features. (tradeoffs such as overfitting, to be discussed) Last layer usually acts as a classifier 
that outputs the class label.


(What is 'end-to-end trainable'?
Let's say you want to make an autonomous vehicle, you have two options, you can either get 
four smaller networks each one dedicated to some sensor and let's say they detect objects, 
then you can have a standard program which controls the car based on the output of the smaller 
neural nets. Alternatively you can feed in the sensor information and have the network regress 
the end control instructions (ie steering angle ). The latter is “end to end”. - Tapa Ghosh
OR basically one could just see it as with a certain input, you are able to determine what the 
exact output might be, whether its a string of words or in the case of the car, the steering angle etc.)

----------------------------------------------------------------------------------------------------------------

#### c) Deep Learning and Neural Networks
|*Perceptron*|*Multi-Layer Perceptrons (MLP)* or ANNs (For more complex problems)|
|------------|-------------------------------------------------------------------|
|Neural Network that only contains one neuron |Consists of: <br/> 1) Input layer,   <br/> 2) Hidden layer,   <br/>3) Weight connections (edges),   <br/>4) Output layer  <br/> <br/>Training process consists of 3 main steps:   <br/>1) Feedforward operation,   <br/>2) Calculate the error,  <br/>3) Error Optimization: use of backpropogation and gradient descent (helps in selecting the most optimum parameters that minimize the error function)|


![ANNs](https://www.extremetech.com/wp-content/uploads/2015/07/NeuralNetwork.png)

First, let's look at a *single perceptron*:  

How does an artificial neuron mimic a biological one?  

<b>i) Weighted sum of inputs</b> calculated to represent the total strength of input signals (weights vector)
(Each input signals/feature (x<sub>i</sub>) are assigned weights (w<sub>i</sub>) -weights assigned to edges-
to reflect the importance of a certain feature. How are weight values assigned?)

z= \sum (x<sub>i</sub>w<sub>i</sub>) + b, where b=bias 

The bias allows for a better fit of a graph onto the dataset (how?).  
The weighted sum function above can also be seen as a linear combination, y=mx+c.  
The bias thus represents the y-intercept, without it the graph would always pass through the origin, and 
hence the line would be a poorer fit.  
Input layer can be given biases by introducing an extra input. Value of the bias is treated as an extra weight 
and is learned and adjusted by the neuron to minimize the cost function.  


<b>ii) Activation function</b> to determine if the resulting sum (from input signal) should 
result in an output of 0 or 1 (Neuron Functions). Output also dependent on Activation Function used.

![Perceptron](/perceptron.png)

<b>Activation Functions(Main few + Explanantions):</b>

- Linear Transfer Function (or Identity Function)  
*activation(z) = z = wx + b*  
Signal passes through the function unchanged (as good as no activation function)  
Composition of two linear functions is still a linear function, so unless theres a non-linear function, 
the model will not be learning anything. (Why? To do with constant gradient and backpropogation)

----------------------------------------------------------------------------------------------------------------

- Heaviside Step Function (Binary)
```
if(w.x+b<=0){
    return 0
}else{
    return 1
}
```

----------------------------------------------------------------------------------------------------------------

- Sigmoid Function (One of the more common activation functions)  
Sigmoid squishes all values toa probability between 0 and 1 (reduces extreme values/outliers)  

sigmoid(z) = 1/(1+e<sup>-z</sup>)  

How is this better than linear graphs?   
Linear graphs with a gradient and no limits can have values below zero 
and above 1 (and hence does not work in most cases).
There is then a need to make the probability range > 0 and < 1, and hence through some form of 
[derivation](https://beckernick.github.io/sigmoid-derivative-neural-network/) we get the sigmoid function

Tensorflow/Keras:
```
keras.activations.sigmoid(x)
```
----------------------------------------------------------------------------------------------------------------


- Softmax Function  
Generalization of the sigmoid function. Used to obtain classification probablities when there are more than 
2 classes. (Forces outputs to sum to 1) - eg. Numbers (10 choices, 0-9)

sigmoid(x<sub>j</sub>) = e<sup>x<sub>j</sub></sup>/sum(i)(e<sup>x<sub>i</sub></sup>)  

!!!- Softmax function is the main function to be used when you need to predict a class between more than 2 classes  
(if only 2 classes it will essentially work as a sigmoid function)

Tensorflow/Keras:
```
keras.layers.Softmax(axis=-1)
```

----------------------------------------------------------------------------------------------------------------

- Hyperbolic Tangent Function (tanh Function)
Shifted version of the sigmoid function (but instead all tanh(x) values are between -1 and 1)  
Works better in the hidden layers as it has the effect of centering data so that the mean of the data is closer 
to 0 than 0.5 (makes learning for next layer easier)

tanh(x) = sinh(x)/cosh(x) = (e<sup>x</sup>-e<sup>-x</sup>)/(e<sup>x</sup>+e<sup>-x</sup>)

Issues with tanh as well as sigmoid functions: If z is very large/small, gradient becomes small and gradient descent 
slows down.

Tensorflow/Keras:
```
keras.activations.tanh(x)
```

----------------------------------------------------------------------------------------------------------------

- Rectified Linear Unit (ReLU) Function (!!!-As of now, one of the better activation functions 
as it works well in many different scenarios)

Function activates a node only if input value >0, else output is always zero.  
If >0, output will have a linear relationship with the output variable f(x) = max(0,x)

```
if(x<0){
    return 0
}else{
    return x
}
```

Tensorflow/Keras:
```
keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
```

Disadvantage: Derivative equals to 0 when x is negative

----------------------------------------------------------------------------------------------------------------

- Leaky ReLU Function
Solves the disadvantage of ReLU Function by introducing a small negative slope when x<0  

f(x)=max(0.01x,x)

```
if(x<0){
    return 0.01x
}else{
    return x
}
```

Tensorflow/Keras:
```
keras.layers.LeakyReLU(alpha=0.3)
```

----------------------------------------------------------------------------------------------------------------

- Rough Guideline as to which activation function to use:  
    - Hidden layers: ReLU or Leaky ReLU (Reduces likelihood for gradient to vanish)  
    - Output layer:  
        - For mutually exclusive classes, Softmax Function is used,  
        - Sigmoid Function for binary classification,  
        - No activation function needed for regression problems  

To explore other Activation functions: elu, selu, prelu, softplus, softsign, hard_sigmoid, exponential, 
ThresholdedReLU

Activation Functions Summary (Adapted from Deep Learning for Vision Systems, references below)
![Activation Functions Summary](/Activation_functions.jpg)




And now, let's move on to some *Multi-Layer Perceptrons*!

- One neuron is not enough to solve complex problems (why?)  
Not all data is linearly separable, which means that the classes cannot be split according to a single straight
line (will result in mispredictions)  
Nonlinear problems: Dataset needs more than one line that forms a shape to split the data

- What are hidden layers? :   
Neurons stacked together, whose inputs/outputs are not seen during the training of 
the model(hence, hidden)  
Early layers detect simple features (straight lines)  
Later layers detect complex features, pattern within patterns etc.  

- Neural Network not fitting the data (underfitting): More layers would have to be added (deeper learning required)
- Overfitting: Too many layers and instead of learning the relationship between features and labels, the network 
instead remembers the images (also more computationally expensive) - performs well on training data but 
very poorly on data not seen during training

- How to know if underfitting/overfitting has occured?  
For overfitting, check for Accuracy of training set against validation set (training set will be very good, 
whereas validation set will result in poor reading)  
For underfitting, may be abit harder to diagnose but if Accuracy of training vs validation set is similar 
but low in value, then it is most likely underfitting 

- What are some different neural network architectures? 
a) Recurrent neural networks (RNNs)
b) Convolutional neural networks (CNNs)
c) Feedforward Neural Network
d) Radial Basis Function Neural Network
e) [THE LIST GOES ON...] -refer to image below for more architectures

![Neural Networks](https://www.digitalvidya.com/wp-content/uploads/2019/01/Image-1-2.png)


Learning Process:  
i)  Feedforward calculations to produce prediction  
ii) Calculate the error  
iii) Backpropagate error and update weights to minimize error

i)








References:
- Mohamed Elgendy ( [Deep Learning for Vision Systems](https://www.manning.com/books/deep-learning-for-vision-systems) )
- Alex Nasli (3D Object Detection from LiDAR Data with Deep Learning)
- Andrew Ng

## Conda environment setup for MacOS

Install [Anaconda for MacOS](https://docs.anaconda.com/anaconda/install/mac-os/)

Initialising a new conda environment and activating the environment
```
conda update conda
conda create -n yourenvname python=x.x anaconda
source activate yourenvname
```

To install additional packages at initialisation stage:
```
conda install -n yourenvname [package]
```

OR, once initialised and in the environment of choice:
```
conda install [package name]
#eg. 
conda install tensorflow
conda install -c conda-forge opencv
```

To activate/deactivate environment
```
conda activate/deactivate [environment name]
```
