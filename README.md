# Poet - Text Generation with Neural Networks

This project implements character-level text generation using different neural network approaches in both TensorFlow and Keras. It can generate text by learning patterns from input text data.

## Project Structure

```
└── txtgen/
    ├── char_nn_keras.py  - Keras implementation using GRU
    ├── char_rnn_tf.py    - TensorFlow implementation using GRU
    ├── utils.py          - Shared utilities for text processing
    ├── ferdosi.txt       - Sample input text
    ├── test.txt          - Test input text
    ├── model.h5          - Saved Keras model
    ├── modeltest.h5      - Test model checkpoint
    ├── weights.hdf5      - Saved model weights
    └── weightstest.hdf5  - Test weights checkpoint
```

## Features

- Character-level text generation using Recurrent Neural Networks (RNN)
- Multiple implementations:
  - Keras with GRU layers
  - TensorFlow with GRU cells
- Configurable hyperparameters:
  - Sequence length
  - Batch size
  - Hidden layer size
  - Number of layers
  - Learning rate
- Text preprocessing utilities
- Model checkpointing and weight saving
- Temperature-based sampling for text generation

## Usage

### Keras Implementation

The Keras implementation (`char_nn_keras.py`) provides a simpler interface with:
- One-hot encoding for character vectors
- GRU-based sequence modeling
- Temperature-controlled text generation

### TensorFlow Implementation

The TensorFlow implementation (`char_rnn_tf.py`) offers more control with:
- Configurable flags for model parameters
- Multi-layer GRU architecture
- TensorBoard support for visualization
- Checkpoint management

### Utilities

The `utils.py` module provides:
- Text preprocessing and vocabulary management
- Batch generation for training
- Sampling functions for text generation

## Requirements

- TensorFlow
- Keras
- NumPy

## Training Data

The model can be trained on any text file. Two sample files are included:
- `ferdosi.txt`: Main training text
- `test.txt`: Test data for validation

## Model Files

The project includes pre-trained models and weights:
- `model.h5`: Main Keras model
- `modeltest.h5`: Test checkpoint
- `weights.hdf5`: Trained weights
- `weightstest.hdf5`: Test weights