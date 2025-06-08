import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import label_binarize

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

import xgboost as xgb
import lightgbm as lgb

# --- Step 0: Load data ---
dataset = pd.read_csv(r"C:\Users\mohap\vscodeproject\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# --- Step 1: Preprocessing function ---
def preprocess_texts(texts, remove_stopwords=True):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    corpus = []
    for review in texts:
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        words = review.split()
        words = [ps.stem(word) for word in words if word not in stop_words]
        corpus.append(' '.join(words))
    return corpus

# --- Step 2: Vectorization & Classification pipeline function ---
def run_classifiers(corpus, y, test_size=0.2, remove_stopwords=True):
    # Choose vectorizers
    vectorizers = {
        'BoW': CountVectorizer(),
        'TF-IDF': TfidfVectorizer()
    }
    
    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42)
    }
    
    results = []
    
    # Split data once
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(corpus, y, test_size=test_size, random_state=42, stratify=y)
    
    for vec_name, vec in vectorizers.items():
        # Fit vectorizer on train only
        X_train = vec.fit_transform(X_train_raw).toarray()
        X_test = vec.transform(X_test_raw).toarray()
        
        for clf_name, clf in classifiers.items():
            print(f"Training {clf_name} with {vec_name}...")
            clf.fit(X_train, y_train)
            
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)
            
            # Try to get probabilities for AUC (some classifiers may not support it well)
            try:
                y_train_prob = clf.predict_proba(X_train)[:,1]
                y_test_prob = clf.predict_proba(X_test)[:,1]
            except:
                # Use decision function for SVM
                if hasattr(clf, 'decision_function'):
                    y_train_prob = clf.decision_function(X_train)
                    y_test_prob = clf.decision_function(X_test)
                else:
                    # fallback: binary predictions (not ideal)
                    y_train_prob = y_train_pred
                    y_test_prob = y_test_pred
            
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            # Binarize labels if not binary already (just in case)
            y_train_bin = label_binarize(y_train, classes=[0,1]).ravel()
            y_test_bin = label_binarize(y_test, classes=[0,1]).ravel()
            
            # Calculate AUC if possible
            try:
                train_auc = roc_auc_score(y_train_bin, y_train_prob)
                test_auc = roc_auc_score(y_test_bin, y_test_prob)
            except:
                train_auc = np.nan
                test_auc = np.nan
            
            # Bias and variance proxy
            bias = 1 - train_acc
            variance = train_acc - test_acc
            
            results.append({
                'Vectorizer': vec_name,
                'Classifier': clf_name,
                'Train Accuracy': train_acc,
                'Test Accuracy': test_acc,
                'Train AUC': train_auc,
                'Test AUC': test_auc,
                'Bias': bias,
                'Variance': variance
            })
    
    return pd.DataFrame(results)

# Step 3: Run pipeline ---

remove_stopwords = True  # Change to False to keep stopwords
corpus = preprocess_texts(dataset['Review'], remove_stopwords=remove_stopwords)
y = dataset.iloc[:,1].values

# Run experiment
df_results = run_classifiers(corpus, y, test_size=0.2, remove_stopwords=remove_stopwords)

# Show sorted by test accuracy or AUC
print(df_results.sort_values(by='Test Accuracy', ascending=False).reset_index(drop=True))

# Optional: plot ROC curve for best model (you can implement based on df_results)
# --- Step 4: Plot ROC Curve for Best Model (by Test AUC) ---

# Find best model based on Test AUC
best_result = df_results.sort_values(by='Test AUC', ascending=False).iloc[0]
best_vectorizer_name = best_result['Vectorizer']
best_classifier_name = best_result['Classifier']

print(f"\n🎯 Best Model: {best_classifier_name} with {best_vectorizer_name} (Test AUC: {best_result['Test AUC']:.2f})")

# Re-instantiate vectorizer and classifier
vectorizer = CountVectorizer() if best_vectorizer_name == 'BoW' else TfidfVectorizer()
classifier_dict = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42)
}
classifier = classifier_dict[best_classifier_name]

# Re-train the model
X_train_raw, X_test_raw, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=42, stratify=y)
X_train = vectorizer.fit_transform(X_train_raw).toarray()
X_test = vectorizer.transform(X_test_raw).toarray()
classifier.fit(X_train, y_train)

# Get probabilities for ROC curve
y_test_prob = classifier.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
auc_score = roc_auc_score(y_test, y_test_prob)

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve: {best_classifier_name} with {best_vectorizer_name}')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
