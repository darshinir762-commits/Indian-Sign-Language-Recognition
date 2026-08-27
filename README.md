# Indian Sign Language Recognition

A deep learning-based web application for recognizing Indian Sign Language (ISL) hand signs from images.

## Overview

This project uses deep learning models to classify Indian Sign Language alphabet gestures into their corresponding letters from A to Z.

The system includes model training, data preprocessing, model evaluation, a FastAPI backend for inference, and a React-based frontend for user interaction.

## Features

- Indian Sign Language alphabet recognition (A–Z)
- Image-based sign prediction
- Multiple deep learning models
- EfficientNet-based prediction
- MobileNet-based prediction
- Top-3 prediction results
- Prediction confidence score
- Inference-time measurement
- FastAPI backend
- React frontend
- Model evaluation and comparison
- Confusion matrices
- Classification reports
- Webcam-based inference support

## Models

The project includes the following models:

| Model | Status |
|---|---|
| CNN | Training and evaluation code included |
| MobileNet | Trained model included |
| EfficientNet | Trained model included |

The trained CNN model is not included in the GitHub repository because of its large file size.

## Project Architecture

```text
User
  │
  ▼
React Frontend
  │
  │ Image Upload / Webcam
  ▼
FastAPI Backend
  │
  ▼
Prediction Service
  │
  ├── CNN
  ├── MobileNet
  └── EfficientNet
  │
  ▼
Prediction Result
  │
  ├── Predicted Letter
  ├── Confidence
  ├── Top-3 Predictions
  └── Inference Time