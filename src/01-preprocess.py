import pandas as pd
import numpy as np
import re

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
    # Check if the password contains a sequence of the same character repeated, e.g., 'aaaa', 'bbbb', '@@@@', etc.
    if re.search(r"(.)\1{2,}", password):  # Checks for any character repeated 3 or more times consecutively
        return 1
    return 0
def caseRatio(row):
    if row['Count(Uppercase)'] == 0:
        return 0
    return row['Count(Lowercase)'] / row['Count(Uppercase)']

dataset = pd.read_csv("pwlds.csv")
dataset['Length'] = dataset['Password'].str.len()
dataset['Password'] = dataset['Password'].fillna("").astype(str)
dataset['Count(Alphabets)'] = dataset['Password'].apply(countAlphabets)
dataset['Count(Numerics)'] = dataset['Password'].apply(countNumbers)
dataset['Count(SpecialChars)'] = dataset['Password'].apply(countSpecialChars)
dataset['Count(Uppercase)'] = dataset['Password'].apply(countUppercase)
dataset['Count(Lowercase)'] = dataset['Password'].apply(countLowercase)
dataset['RepeatedChars'] = dataset['Password'].apply(hasRepeatedCharacters)
dataset['CaseRatio'] = dataset.apply(caseRatio, axis=1)
strengthColumn = dataset.pop('Strength')
dataset['Strength'] = strengthColumn
print(dataset)

dataset.to_csv("New-Featured-Dataset.csv", index=False)
