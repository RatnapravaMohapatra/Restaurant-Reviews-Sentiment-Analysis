# 📝 Restaurant Reviews Sentiment Analysis

This project is a comprehensive machine learning pipeline for **sentiment analysis** on restaurant reviews. It leverages **Natural Language Processing (NLP)** techniques and evaluates multiple classifiers using both **Bag of Words (BoW)** and **TF-IDF** vectorization.

## 🔍 Objective

To classify customer reviews as **positive (1)** or **negative (0)** based on their text content and compare different ML models based on accuracy, AUC score, bias, and variance.

---

## 📦 Features

- ✅ Text preprocessing using NLTK:
  - Cleaning text (removing special characters)
  - Lowercasing
  - Tokenization
  - Stopword removal (optional)
  - Stemming
- ✅ Feature extraction using:
  - Bag of Words
  - TF-IDF
- ✅ Model evaluation metrics:
  - Accuracy (Train/Test)
  - ROC-AUC Score
  - Bias & Variance Proxy
- ✅ Classifier comparison:
  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - Naive Bayes
  - XGBoost
  - LightGBM
- ✅ ROC Curve visualization for best model

---

## 📁 Dataset

The dataset used is [`Restaurant_Reviews.tsv`](path/to/file), a collection of customer reviews and their corresponding sentiment labels.  
Format:  
- `Review` (text)  
- `Liked` (0 = Negative, 1 = Positive)

---

## 📊 Results Summary

After comparing classifiers using both vectorization methods, the best model is selected based on **Test AUC Score**, and its ROC curve is plotted.

Example output table:

| Vectorizer | Classifier         | Test Accuracy | Test AUC | Bias | Variance |
|------------|--------------------|---------------|----------|------|----------|
| TF-IDF     | XGBoost            | 0.88          | 0.94     | 0.12 | 0.03     |
| BoW        | Logistic Regression| 0.86          | 0.91     | 0.14 | 0.04     |

---

## 📈 ROC Curve Example

![ROC Curve](path/to/roc_curve.png)

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/restaurant-sentiment-analysis.git
   cd restaurant-sentiment-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```bash
   python sentiment_analysis.py
   ```

---

## 🔧 Requirements

- Python 3.7+
- pandas, numpy, matplotlib
- scikit-learn
- nltk
- xgboost
- lightgbm

> Make sure to download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```
