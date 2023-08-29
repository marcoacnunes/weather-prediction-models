# Weather Prediction Models 🌦️

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
- [Models and Evaluation](#models-and-evaluation)
- [Results](#results)

## Introduction

This project aims to predict whether it will rain tomorrow based on historical weather data. We use various machine learning models including Linear Regression, K-Nearest Neighbors (KNN), Decision Trees, and Logistic Regression.

## Dataset 📊

The dataset is sourced from the Corgis project and can be found [here](https://corgis-edu.github.io/corgis/datasets/csv/weather/weather.csv). It provides various metrics like wind direction, precipitation, and more.

## Getting Started 🚀

### Prerequisites

Ensure you have the following libraries installed:
- `pandas`
- `numpy`
- `sklearn`
- `requests`

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/marcoacnunes/weather-prediction-models.git
   ```
2. Install the required packages:
   ```bash
   pip install pandas numpy sklearn requests
   ```

### Usage

Run the main script to train the models and evaluate their performance:
```bash
python main.py
```

## Models and Evaluation 🤖

### 1. Linear Regression
- **Purpose**: Predict the amount of precipitation.
- **Metrics**: MAE, MSE, R^2 Score.

### 2. K-Nearest Neighbors (KNN)
- **Purpose**: Classify whether it will rain tomorrow.
- **Metrics**: Accuracy, Jaccard Index, F1 Score.

### 3. Decision Trees
- **Purpose**: Classify whether it will rain tomorrow.
- **Metrics**: Accuracy, Jaccard Index, F1 Score.

### 4. Logistic Regression
- **Purpose**: Classify whether it will rain tomorrow.
- **Metrics**: Accuracy, Jaccard Index, F1 Score, Log Loss.

## Results 📈

| Model               | Accuracy   | Jaccard Index | F1 Score  | Log Loss  |
|---------------------|------------|---------------|----------|----------|
| Linear Regression   | 99.79% (R^2 Score) | -        | -        | -        |
| KNN                 | 82.23%     | 23.02%        | 37.43%   | -        |
| Decision Tree       | 86.02%     | 43.68%        | 60.80%   | -        |
| Logistic Regression | 99.88%     | 99.39%        | 99.69%   | 0.0618   |
| SVM                 | 80.56%     | 0.00%         | 0.00%    | -        |

