# 3D Self-Organizing Map

- [Japanese](./README.md)
- [English](./README.en.md)

This Python program implements a 3D Self-Organizing Map (SOM) based on the concept introduced by Teuvo Kohonen in the 1980s.  
A SOM is a type of artificial neural network trained using unsupervised learning to map the input space of training samples into a lower-dimensional space (usually two-dimensional).  
In this implementation, we extend the traditional SOM into three dimensions.

## Installation

After installing Python 3.9 or later, follow these steps to install:

```bash
$ git clone git@github.com:c19yamahar/SOM3D.git
$ cd SOM3D
$ pip install -r requirements.txt
```

## Usage

After importing the SOM3D class, initialize the SOM with your desired grid dimensions, input dimension, learning rate, and sigma value. Call the train method to train the SOM with your dataset, and use the winner method to find the winning node of the SOM for a given input vector. You can retrieve the vector for a specific coordinate using the get_vector method.

```python
import numpy as np
from som3d import SOM3D

# Initialize the SOM with a 10x10x10 grid and an input dimension of 3

som = SOM3D(x_dim=10, y_dim=10, z_dim=10, input_dim=3)

# Generate random training data

training_data = np.random.rand(100, 3)

# Train the SOM

som.train(training_data, n_epoch=1000)

# Find the winning node for a given input vector

input_vector = np.array([0.5, 0.5, 0.5])
winner_coordinates = som.winner(input_vector)
print(f"Winning node coordinates: {winner_coordinates}")

# Get the vector for a specific coordinate

vector = som.get_vector((5, 5, 5))
print(f"Vector at coordinates (5, 5, 5): {vector}")

```
