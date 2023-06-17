# AI-generated image detector, COMP4107-Project

This project has several parts for creating an AI-generated image detector, that were used to experiment and report on model parameters that increased model effectiveness and accuracy.

## Dataset generator

Python script generates a local custom-format dataset that pulls from various sources such as DiffusionDB for AI-generated images (specifically using Stable Diffusion) and LAION artistic datasets for non-AI generated images to build a dataset for a model to train on.

## Model generator

Python script can train a 2D convolutional neural network on the generated dataset, with the optional help of different available pre-processing techniques such as edge detection and corner detection.
