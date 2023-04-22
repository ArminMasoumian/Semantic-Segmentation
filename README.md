# Semantic-Segmentation

This repository contains Python code for a semantic segmentation model using TensorFlow. The semantic segmentation model is a deep learning model that can be used to segment images into different classes. The model uses convolutional neural networks (CNNs) to learn the features of the image and then uses a decoder network to generate a segmentation map for each pixel of the input image.

## Requirements

* Python 3.6 or later

* TensorFlow 2.0 or later

* NumPy

## Files

* `model.py`: Defines the `SemanticSegmentationModel` class for the semantic segmentation model.

* `train.py`: Script to train the semantic segmentation model on training data.

* `utils.py`: Contains utility functions to load the training and validation data.


## Usage

1. Install the required packages by running `pip install -r requirements.txt`.

2. Prepare your training and validation data as numpy arrays and save them to disk as `x_train.npy`, `y_train.npy`, `x_val.npy`, and `y_val.npy`.

3. Train the model by running `python train.py`.

4. The trained model will be saved to a file called `model.h5`.

## Example

Here is an example of how to use the `SemanticSegmentationModel` class to define and compile the model:

```python
import tensorflow as tf
from model import SemanticSegmentationModel

# Define the model
input_shape = (256, 256, 3)
num_classes = 2
model = SemanticSegmentationModel(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

```

## Results

![](https://github.com/ArminMasoumian/Semantic-Segmentation/blob/main/Semantic%20Segmentation.jpg)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ArminMasoumian/Semantic-Segmentation/blob/main/LICENSE) file for details.
