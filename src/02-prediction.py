import torch
import random
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class PasswordNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PasswordNN, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.1)        
        self.hidden2 = nn.Linear(128, 64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.1)
        self.hidden3 = nn.Linear(64, 32)
        self.batchnorm3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.1)
        self.output = nn.Linear(32, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.batchnorm1(self.hidden1(x)))  # BatchNorm + ReLU
        x = self.dropout1(x)
        x = torch.relu(self.batchnorm2(self.hidden2(x)))  # BatchNorm + ReLU
        x = self.dropout2(x)
        x = torch.relu(self.batchnorm3(self.hidden3(x)))  # BatchNorm + ReLU
        x = self.dropout3(x)
        x = self.output(x)
        return torch.softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device selected: {device}')

model = PasswordNN(input_dim=8, output_dim=5).to(device)
model.load_state_dict(torch.load("PasswordNN_v2.pth", weights_only=True))

model.eval()
print("Model loaded successfully.")


import torch
import re

def getLength(password):
    return len(password)
def countAlphabets(password):
    return sum(char.isalpha() for char in password)
def countNumbers(password):
    return sum(char.isdigit() for char in password)
def countSpecialChars(password):
    return sum(not char.isalnum() for char in password)
def countUppercase(password):
    return sum(char.isupper() for char in password)
def countLowercase(password):
    return sum(char.islower() for char in password)
def hasRepeatedCharacters(password):
    return 1 if re.search(r"(.)\1{2,}", password) else 0
def caseRatio(uppercase_count, lowercase_count):
    if uppercase_count == 0:
        return 0
    return lowercase_count / uppercase_count

def preprocess_input(password):
    # Extract features
    length = getLength(password)
    alphabets = countAlphabets(password)
    numbers = countNumbers(password)
    special_chars = countSpecialChars(password)
    uppercase = countUppercase(password)
    lowercase = countLowercase(password)
    repeated_chars = hasRepeatedCharacters(password)
    ratio = caseRatio(uppercase, lowercase)
    
    # Create the feature list
    features = [
        length,
        alphabets,
        numbers,
        special_chars,
        uppercase,
        lowercase,
        repeated_chars,
        ratio
    ]
    
    # Convert features to a PyTorch tensor and move it to the selected device
    input_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    return input_tensor

def test_model_with_input(user_input):
    model.to(device)
    input_tensor = preprocess_input(user_input)
    input_tensor = input_tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        print(output)
        _, predicted_class = torch.max(output, 1)
        predicted_label = predicted_class.item()
        print(f"Predicted password strength: {predicted_label}")


test_model_with_input("7hqwv")
test_model_with_input("7hqwaskldjv")
test_model_with_input("7hqwas123123kldj!@#v")
