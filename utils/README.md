# Titanic Survival Classification Dashboard

## Problem Statement
The goal of this project is to build an interactive dashboard that predicts passenger survival from the Titanic disaster, based on demographic and ticket information. Users will be able to adjust classification parameters and observe their impact on model performance.

## Dataset
- **Name:** Titanic Passenger Dataset
- **Source:** Kaggle (https://www.kaggle.com/c/titanic/data)
- **Target Variable:** Survived (1 = survived, 0 = did not survive)
- **Features Available:**
  - PassengerId
  - Pclass (Ticket class: 1 = 1st, 2 = 2nd, 3 = 3rd)
  - Name
  - Sex
  - Age
  - SibSp (Number of siblings / spouses aboard)
  - Parch (Number of parents / children aboard)
  - Ticket
  - Fare
  - Cabin
  - Embarked (Port of Embarkation)

- **Selected Features for Initial Model:**
  - Age (numeric)
  - Fare (numeric)

## Algorithm
I will apply a **Support Vector Machine (SVM)** classifier for this project. SVM is well-suited for binary classification tasks and will help us visualize decision boundaries effectively.

## Dashboard
The dashboard will:
- Allow tweaking of SVM parameters (kernel type, C, gamma, etc.)
- Visualize decision boundaries
- Display accuracy and performance metrics
- Optional: show confusion matrix and feature importance

## Usage
Run the app with `python app.py`. Use the interactive controls to explore how different SVM parameters affect Titanic survival predictions.
