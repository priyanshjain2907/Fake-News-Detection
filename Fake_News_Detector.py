# =============================
# 1. IMPORT LIBRARIES
# =============================
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report

# =============================
# 2. LOAD DATA
# =============================
fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# =============================
# 3. ADD LABELS
# =============================
fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# =============================
# 4. COMBINE + SHUFFLE
# =============================
data = pd.concat([fake_df, true_df], axis=0)
data = data.sample(frac=1, random_state=42)

# =============================
# 5. TEXT CLEANING FUNCTION
# =============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Combine title + text + clean
data["content"] = (data["title"] + " " + data["text"]).apply(clean_text)

X = data["content"]
y = data["label"]

# =============================
# 6. TRAIN TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 7. VECTORIZATION (N-GRAMS)
# =============================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =============================
# 8. MODEL 1: LOGISTIC REGRESSION
# =============================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)

lr_pred = lr_model.predict(X_test_vec)

print("🔹 Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# =============================
# 9. MODEL 2: PASSIVE AGGRESSIVE
# =============================
pac_model = PassiveAggressiveClassifier(max_iter=1000)
pac_model.fit(X_train_vec, y_train)

pac_pred = pac_model.predict(X_test_vec)

print("🔹 Passive Aggressive Accuracy:", accuracy_score(y_test, pac_pred))
print(classification_report(y_test, pac_pred))

# =============================
# 10. FEATURE IMPORTANCE
# =============================
feature_names = vectorizer.get_feature_names_out()
coefficients = lr_model.coef_[0]

top_fake = sorted(zip(coefficients, feature_names))[:10]
top_real = sorted(zip(coefficients, feature_names), reverse=True)[:10]

print("\n🔥 Top words for FAKE:")
for coef, word in top_fake:
    print(word)

print("\n🔥 Top words for REAL:")
for coef, word in top_real:
    print(word)

# =============================
# 11. PREDICTION FUNCTION
# =============================
def predict_news(news_text):
    news_text = clean_text(news_text)
    vec = vectorizer.transform([news_text])
    
    prediction = lr_model.predict(vec)[0]
    prob = lr_model.predict_proba(vec).max()
    
    return prediction, prob

# =============================
# 12. TEST PREDICTION
# =============================
news = "Breaking: Government secretly planning massive tax increase."

label, confidence = predict_news(news)

print("\n🧪 Test Prediction:")
print("Prediction:", label)
print("Confidence:", round(confidence, 2))

# =============================
# 13. SAVE MODEL
# =============================
import pickle

pickle.dump(lr_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))