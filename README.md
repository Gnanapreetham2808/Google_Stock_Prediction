# 📈 Google Stock Price Prediction Using LSTM

This project explores the application of Long Short-Term Memory (LSTM) networks in forecasting Google’s stock prices using historical data. We compare different models including ARIMA, basic LSTM, and hybrid deep learning architectures to assess their accuracy and effectiveness in time series forecasting.

---



## 📌 Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Relevance](#relevance)
- [Objective](#objective)
- [Problem Statement](#problem-statement)
- [Literature Review](#literature-review)
- [Methodology](#methodology)
  - ARIMA Model
  - Basic LSTM Model
  - Proposed Hybrid Model
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

---

## 📖 Introduction
The stock market is dynamic and influenced by various factors, making price prediction a complex task. LSTM networks are well-suited for this challenge due to their ability to learn long-term dependencies in sequential data.

---

## 🎯 Motivation
Machine learning and deep learning are revolutionizing financial forecasting. LSTM networks provide a powerful framework to predict stock prices based on historical data.

---

## 📈 Relevance
Accurate predictions can provide a significant edge in the financial market. This project is highly relevant in today’s AI-driven environment.

---

## 🎯 Objective
- Develop an LSTM-based model for predicting Google’s stock prices.
- Compare performance with traditional models (e.g., ARIMA).
- Evaluate prediction accuracy using standard metrics like MAE, MSE, RMSE.
- Explore the benefits of hybrid models (CNN+LSTM, GRU+CNN, Bi-LSTM+CNN).

---

## ❓ Problem Statement
Traditional stock price prediction models struggle with non-linear and noisy data. This project aims to overcome these limitations using LSTM-based models to forecast Google stock prices more accurately.

---

## 📚 Literature Review
| Sl. No | Title | Methodology | Dataset | Performance Metrics | Advantages | Disadvantages |
|--------|-------|-------------|---------|----------------------|------------|----------------|
| 1 | Optimized Deep LSTM (ARO) | LSTM + Artificial Rabbits Optimization | DJIA (2018–2023) | MSE, MAE, MAPE, R² | Higher accuracy | High training cost |
| 2 | CNN-LSTM with Leading Indicators | CNN + LSTM + Futures & Options | 10 US & Taiwanese Stocks | Accuracy, MSE | Incorporates external indicators | High complexity |
| 3 | MMLSTM | Multivariate LSTM + Sentiment Analysis | AAPL, Tweets, News | MAE, MAPE | High accuracy, diverse data | Sensitive to sentiment errors |
| 4 | LSTM + BERT | Sentiment (BERT) + LSTM | Chinese Stocks | MAE, RMSE | Combines emotion + data | Lower accuracy on some stocks |
| 5 | MLS LSTM | Multi-layer LSTM + Adam Optimizer | Samsung (2016–2021) | MAPE 2.18%, RMSE 0.028 | 98.1% accuracy | High compute demand |
| 6 | Word2Vec + LSTM | News Headlines + LSTM | Reuters News | Accuracy ~65% | Combines text + time-series | Varies across companies |

---

## ⚙️ Methodology

### 1️⃣ ARIMA Model
- Traditional time-series forecasting model
- Parameters: (p, d, q)
- Metrics: MAE, RMSE

### 2️⃣ Basic LSTM Model
- Single-layer LSTM with dropout
- Input: 60 days of closing prices
- Output: Prediction for the next day

### 3️⃣ Proposed Hybrid Model
- Models: CNN+LSTM, GRU+CNN, Bi-LSTM+CNN
- Techniques:
  - Data Scaling using MinMaxScaler
  - Hyperparameter tuning via GridSearchCV
  - Multiple dropout layers to prevent overfitting

---

## 📊 Results

### 🔹 ARIMA
- Captures linear trends
- Metrics: MAE, RMSE

### 🔹 Basic LSTM
- Captures long-term dependencies
- Metrics: MAE, RMSE, Training vs Validation Loss

### 🔹 Proposed Hybrid Model
- Best accuracy using Bi-LSTM + CNN
- Metrics: MSE, Validation Accuracy, Dropout Performance
- Handles both spatial and temporal patterns in data

---

## ✅ Conclusion
LSTM-based models, especially hybrid variants, outperform traditional models like ARIMA in predicting Google stock prices. Incorporating sentiment and external indicators further improves forecasting capabilities. The project demonstrates how deep learning models can be leveraged for financial decision-making and provides a roadmap for more robust, real-world applications.

---

## 📚 References
1. Chandramohan, R., & Jothi, K. (2021). *Stock Price Prediction Using LSTM Neural Network.*
2. Khan, M. A., & Qureshi, A. (2021). *A Hybrid LSTM Model for Stock Price Prediction.*
3. Nassif, A. B., & Zaki, S. M. (2022). *Stock Market Prediction Using LSTM and ML Techniques.*
4. Jiang, H., & Wang, S. (2020). *A Hybrid LSTM Model for Stock Price Prediction.*
5. Kou, G., & Xu, Y. (2020). *A Deep Learning Approach for Stock Price Prediction.*
6. Sahu, P. K., & Sahu, N. K. (2023). *Stock Price Prediction Using LSTM: A Review.*

---

## 📎 Appendix
- Tools Used: Python, TensorFlow/Keras, NumPy, Matplotlib, Sklearn
- IDE: Jupyter Notebook / Google Colab
- Data Source: Yahoo Finance (Google Stock Prices)


