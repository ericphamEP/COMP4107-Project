# AI-generated image detector

This school project has several parts for creating an AI-generated image detector, that were used to experiment and report on model parameters that increased model effectiveness and accuracy.

## Dataset generator

Python script generates a local custom-format dataset that pulls from various sources such as DiffusionDB for AI-generated images (specifically using Stable Diffusion) and LAION artistic datasets for non-AI generated images to build a dataset for a model to train on.

## Model generator

Python script can train a 2D convolutional neural network Keras model on the generated dataset, with the optional help of different available pre-processing techniques such as edge detection and corner detection.

## Development

These source files use Python and various libraries such as Numpy, pandas, Pillow, and TensorFlow. Some preprocessing options also use OpenCV.
