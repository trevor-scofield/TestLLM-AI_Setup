from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x/255).astype('float32')
y = to_categorical(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

"""
Skeleton plan for an LLM training workflow.

This file is intentionally minimal and comment-driven.
Use it as a roadmap for the next implementation steps.
"""


def load_data():
    """
    TODO: Load your dataset here.
    Steps to consider:
    - Choose the training corpus or dataset.
    - Download or load the data locally.
    - Prepare the input format expected by your model.
    """
    pass


def preprocess_data():
    """
    TODO: Clean and format the data.
    Steps to consider:
    - Tokenize text or prepare embeddings.
    - Normalize values if needed.
    - Split into train, validation, and test sets.
    """
    pass


def build_model():
    """
    TODO: Define the model architecture.
    Steps to consider:
    - Choose the model type.
    - Add layers, embeddings, or transformer blocks.
    - Set optimizer, loss, and evaluation metrics.
    """
    pass


def train_model():
    """
    TODO: Train the model.
    Steps to consider:
    - Feed training data into the model.
    - Use validation data to monitor progress.
    - Save checkpoints or training history if needed.
    """
    pass


def evaluate_model():
    """
    TODO: Evaluate the trained model.
    Steps to consider:
    - Measure accuracy, loss, or other relevant metrics.
    - Inspect outputs and identify weaknesses.
    - Decide whether to refine the dataset or model.
    """
    pass


def main():
    """
    TODO: Connect the workflow in order.
    Suggested order:
    1. Load data
    2. Preprocess data
    3. Build model
    4. Train model
    5. Evaluate model
    """
    pass


# End of skeleton.
# Continue by filling in each section with real implementation step by step.
