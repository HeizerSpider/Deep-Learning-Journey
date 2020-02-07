# Deep-Learning-Journey

1) Deep Learning Theory (Continuous update)
2) Conda environment setup for MacOS (Final)

## Deep Learning Theory

### Overview of Deep Learning
Artificial neural networks are a biomimicry of the human brain and its neurons, with both having input signals, a flow of information and outputs.
|Human|AI|
|-----|--|
|Eyes|Camera|
|Neurons|Nodes|

These sensors are not limited to cameras only, but also include lidar and radar sensors which can used as inputs to the model.
In the case of Autonomous Vehicles, depending on the company (mostly), a different variety of sensors are used for the purpose of mapping and localisation.
Some of these sensors include IMUs, GPS and ultrasonic sensors.
However for the purpose of deep learning and computer vision, the main focus will be on cameras/image input.


### Computer Vision Pipeline
1) Image input
2) Pre-Processing
3) Feature extraction
4) ML Model

1) Image input
Images can be interpreted as a function of two variables, x and y (2 dimensional area), ie image is divided into grids (pixels), and each of these pixels 
are assigned a number between 0 to 255 that represents the intensity/brightness of that pixel.

Colour images have 3 values (all between 0-255)-Red, Green, Blue (3 dimensions-(x, y, 3))
Colour is not always important so do be mindful of the purpose of the model and use/prepare the datasets to minimize the amount of time to train the model (reducing complexity)


2) Pre-processing
Essentially preparing the dataset before training/testing the model
- Standardizing the images
- Transforming the colors: Reduce complexity (mentioned above)
- Data augmentation: Augmenting datasets with different version of the intial images (scaling/rotatingflipping, de-texture/colourize etc.) meant to enlarge dataset and exposes neural networks to a greater variety of images and hence able to recognize objects in multiple forms.
- Other Techniques: Can be dependent on the need of the project (removing background, increasing/decreasing noise etc.)

3) Feature extraction
CORE COMPONENT
Purpose: 
- extract useful features for defining/distinguishing an object in an image properly
(eg. car: useful features include wheels, door, spoiler etc.)
- transform the data into features vector-list of features (for learning algorithm to learn the characteristics of the object)
- features must be useful (defining features that allow for the return of a higher probability to that given class through the use of such a feature-quite intuitive, just like how we humans would distinguish between different objects)



Adapted from:
- Mohamed Elgendy (Deep Learning for Vision Systems)
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
