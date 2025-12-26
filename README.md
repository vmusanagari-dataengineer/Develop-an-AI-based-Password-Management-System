# Develop-an-AI-based-Password-Management-System

## Overview
This project focuses on password security using machine learning:
1) Password strength classification (supervised learning)
2) Strong password generation using a Variational Autoencoder (VAE)

The work explores the cybersecurity impact of GenAI and also builds a practical ML pipeline for password strength prediction and generation.

## Dataset
- Stage 1: ROCKYOU + PWNED (labeling attempted using clustering)
- Stage 2: PWLDS dataset with 5-class labels:
  - 0: Very Weak, 1: Weak, 2: Normal, 3: Strong, 4: Very Strong

## Feature Engineering
Extracted features include:
- Password length
- Count of alphabets, numerals, special characters
- Uppercase/lowercase counts and ratios
- Repeated characters

## Model 1: Neural Network Classifier (PyTorch)
Architecture (high level):
- Dense layers: 128 → 64 → 32
- BatchNorm + ReLU + Dropout(0.1)
- Softmax output

Training:
- Optimizer: Adam
- Batch size: 2048
- Epochs: 20
- Learning rate: 0.001

## Model 2: Variational Autoencoder (VAE)
Used for generating strong passwords. Training reduced loss from ~92 to ~1.

## Results
- Validation accuracy: ~94%
- Test accuracy: ~93.7%
- Model performs best on extreme classes (very weak / very strong), and struggles more with intermediate classes.

## Report
See the full paper in: `report/Final_Report_MS-2.pdf`

## How to Run 
1. Create environment
2. Run feature extraction
3. Train classifier
4. Train VAE
5. Generate passwords
