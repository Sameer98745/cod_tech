# spam_detector_codtech.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("codetech/spam.csv", encoding='latin-1')[['v1', 'v2']]

df.columns = ['label', 'message']

# Preprocess the data
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict a new message
sample = ["Congratulations! You've won a free ticket. Call now!"]
sample_vec = vectorizer.transform(sample)
print("\nSample Prediction:", "Spam" if model.predict(sample_vec)[0] == 1 else "Ham")
