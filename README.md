# Deep-Learning-Journey

1) Deep Learning Theory (Continuous update)
2) Conda environment setup for MacOS (Final)

## Deep Learning Theory

Artificial neural networks are a biomimicry of the human brain and its neurons, with both having input signals, a flow of information and outputs.

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
