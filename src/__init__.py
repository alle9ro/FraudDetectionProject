"""
Fraud Detection Project - Source Package

This package contains modules for data preprocessing, model building, training, and evaluation
for a fraud detection system using the Credit Card Fraud Detection dataset from Kaggle.
"""
from .data_preprocessing import load_data, preprocess_data
from .models import build_random_forest, build_neural_network
from .train import train_random_forest, train_neural_network
from .evaluate import evaluate_model