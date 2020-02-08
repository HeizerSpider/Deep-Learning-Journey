# Deep-Learning-Journey

1) Deep Learning Theory (Continuous update)
2) Conda environment setup for MacOS (Final)

## Deep Learning Theory

### a) Overview of Deep Learning Theory
### b) Computer Vision Pipeline
### c) Deep Learning and Neural Networks

#### a) Overview of Deep Learning
Artificial neural networks are a biomimicry of the human brain and its neurons, 
with both having input signals, a flow of information and outputs. Might be good to look to 
the human brain to improve the way the Aritificial Neural Networks (ANNs) are made.  

|Human|AI|
|-----|--|
|Eyes|Camera|
|Neurons|Nodes|

These sensors are not limited to cameras only, but also include [lidar](https://medium.com/@SmartLabAI/3d-object-detection-from-lidar-data-with-deep-learning-95f6d400399a) 
and radar sensors (3d object detection) which can used as inputs to the model.
In the case of Autonomous Vehicles, depending on the company (mostly), a different 
variety of sensors are used for the purpose of mapping and localisation.
Some of these sensors include IMUs, GPS and ultrasonic sensors.
However for the purpose of deep learning and computer vision, the main focus will be on cameras/image input.


#### b) Computer Vision Pipeline

##### i) Image input
##### ii) Pre-Processing
##### iii) Feature extraction
##### iv) Classifier/ML Model

i) Image input  
Images can be interpreted as a function of two variables, x and y (2 dimensional area), 
ie image is divided into grids (pixels), and each of these pixels 
are assigned a number between 0 to 255 that represents the intensity/brightness of that pixel.

Colour images have 3 values (all between 0-255)-Red, Green, Blue (3 dimensions-(x, y, 3))  
Colour is not always important so do be mindful of the purpose of the model and 
use/prepare the datasets to minimize the amount of time to train the model (reducing complexity)


ii) Pre-processing  
Essentially preparing the dataset before training/testing the model  
- Standardizing the images . 
- Transforming the colors: Reduce complexity (mentioned above)  
- Data augmentation: Augmenting datasets with different version of the intial images 
(scaling/rotatingflipping, de-texture/colourize etc.) meant to enlarge dataset and 
exposes neural networks to a greater variety of images and hence able to recognize 
objects in multiple forms.  
- Other Techniques: Can be dependent on the need of the project (removing background, 
increasing/decreasing noise etc.)  

iii/iv) Feature extraction and Classifier  
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

Old ML method (manual hand crafted feature set: HOG, HAAR cascades, SIFT, SURF)  
Input -> Feature Extraction (using abovementioned feature sets) -> Learning Algorithm(SVM/Adaboost) -> Output  

Deep learning on the other hand automatically extracts features (how???)
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

#### c) Deep Learning and Neural Networks
Perceptron: Neural Network that only contains one neuron  

For more complex problems: Multi-Layer Perceptrons (MLP) or ANNs
Consists of 
1) Input layer,   
2) Hidden layer,   
3) Weight connections,   
4) Output layer  

Training process consists of 3 main steps:   
1) Feedforward operation,   
2) Calculate the error,  
3) Error Optimization: use of backpropogation and gradient descent (helps in selecting the most 
optimum parameters that minimize the error function)




References:
- Mohamed Elgendy (Deep Learning for Vision Systems)
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
