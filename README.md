# Stock Prediction (S&P 500)

This project uses machine learning to predict the direction of the S&P 500 index using historical data. The model predicts whether the S&P 500 will go up or down on the next day based on various technical indicators, including stock price and volume data.

Table of Contents:
* [Overview](#overview)
* [Installlation](#installlation)
* [Dependencies](#dependencies)
* [Data Source](#data-source)
* [Model](#model)
* [Prediction](#prediction)
* [Results](#results)
  
## Overview
The S&P 500 index is a stock market index that measures the performance of 500 large publicly traded companies in the U.S. This project aims to predict whether the price of the S&P 500 will increase or decrease the following day using a Random Forest Classifier model trained on historical price data.

The workflow of the project includes:

- Fetching historical data using the Yahoo Finance API (yfinance).
- Preprocessing the data to calculate the target variable (whether the price will increase or decrease the next day).
- Using a Random Forest Classifier to predict the target.
- Evaluating model performance using precision score and backtesting over rolling windows.
## Installation

To run this project, ensure you have Python 3.x installed. You can set up a virtual environment and install the required dependencies as follows:

```bash
git clone https://github.com/karandahal/Stock-Prediction-S-P500.git
cd Stock-Prediction-S-P500
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```
## Dependencies
The following dependencies are required to run this project:

- yfinance: To fetch historical stock data from Yahoo Finance.
- pandas: For data manipulation.
- scikit-learn: For machine learning and model evaluation.
- matplotlib: For data visualization.

```python
pip install yfinance pandas scikit-learn matplotlib
```
## Data Source
The data used in this project is obtained from Yahoo Finance using the yfinance library. It fetches the historical S&P 500 index data by querying ^GSPC.
```python
import yfinance as yf
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
```
## Model
The machine learning model used is a Random Forest Classifier. The classifier is trained to predict whether the S&P 500 index will increase or decrease the next day based on the following features:
- Close: The closing price of the S&P 500 index.
- Volume: The trading volume of the S&P 500 index.
- Open: The opening price of the S&P 500 index.
- High: The highest price during the day.
- Low: The lowest price during the day.
## Prediction

The model uses the historical data to predict a binary target (1 for an increase, 0 for a decrease) for the next day. It uses the following steps:

### Data Preprocessing:
- We remove any irrelevant columns and calculate the target (next day's movement).
- Features such as moving averages and trends over different time horizons (e.g., 2, 5, 60, 250, 1000 days) are computed and used as predictors.

### Model Training:
- The `RandomForestClassifier` is trained on the features from the historical data.
- A rolling window approach is used for backtesting, where the model is trained on historical data up to a certain point and then tested on the next step.

### Evaluation:
- The model's performance is evaluated using the precision score.

```python
from sklearn.metrics import precision_score
preds = model.predict(test[predictors])
precision_score(test["Target"], preds)
```
## Results
The model outputs predictions for the direction of the S&P 500 index (either an increase or a decrease). The evaluation metrics and the confusion matrix give insight into the model's performance, and the backtesting results show how well the model performs over time.

Precision Score is used as a metric to evaluate the model's accuracy in predicting the target direction of the S&P 500. The model performs better with refined features, such as rolling averages and trend analysis.
