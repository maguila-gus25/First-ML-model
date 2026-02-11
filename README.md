# First ML Model - Linear Regression

A simple machine learning project that uses linear regression to predict student scores based on hours studied.

## Overview

This project demonstrates a basic machine learning workflow using scikit-learn to build a linear regression model. The model learns the relationship between study hours and exam scores, then makes predictions and visualizes the results.

## Features

- **Data Preparation**: Creates a dataset with hours studied and corresponding scores
- **Model Training**: Trains a linear regression model using scikit-learn
- **Model Evaluation**: Calculates Mean Squared Error (MSE) to assess model performance
- **Visualization**: Displays a scatter plot with the regression line showing actual vs predicted scores

## Requirements

- Python 3.6 or higher
- Required packages (see `requirements.txt`):
  - numpy
  - pandas
  - scikit-learn
  - matplotlib

## Installation

1. Clone or download this repository

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script:
```bash
python firstmodel.py
```

The script will:
1. Load and prepare the data
2. Split the data into training and testing sets
3. Train a linear regression model
4. Make predictions on the test set
5. Print predictions and Mean Squared Error
6. Display a visualization showing actual scores and the regression line

## Output

The script outputs:
- **Predictions**: Predicted scores for the test set
- **Mean Squared Error**: A metric indicating the model's accuracy (lower is better)
- **Visualization**: A plot showing:
  - Blue dots: Actual scores
  - Red line: Predicted regression line

## Model Details

- **Algorithm**: Linear Regression
- **Features**: Hours Studied (input)
- **Target**: Score (output)
- **Train/Test Split**: 80% training, 20% testing
- **Random State**: 42 (for reproducibility)

## Project Structure

```
First-ML-model/
├── firstmodel.py      # Main script with ML model
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Learning Outcomes

This project demonstrates:
- Data preparation with pandas
- Train-test splitting
- Model training with scikit-learn
- Model evaluation metrics
- Data visualization with matplotlib
