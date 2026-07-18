
import pandas as pd
import numpy as np
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

data = {
    "Review": [
        "This product is amazing and works perfectly",
        "I really loved the quality",
        "Excellent service and fast delivery",
        "Very satisfied with my purchase",
        "The item is fantastic",
        "Best product I have ever bought",
        "Highly recommended",
        "Good quality and affordable price",
        "I am very happy with this product",
        "Absolutely worth the money",
        "The package arrived on time",
        "Customer support was very helpful",
        "Five stars for this product",
        "The quality exceeded my expectations",
        "Very comfortable and easy to use",
        "I will definitely buy again",
        "Everything was perfect",
        "Very reliable product",
        "The design is beautiful",
        "Great experience overall",
        "Very disappointed with this product",
        "The quality is very poor",
        "Waste of money",
        "It stopped working after one day",
        "Worst purchase ever",
        "I hate this product",
        "Very bad customer service",
        "The delivery was too late",
        "Completely useless",
        "Not worth the price",
        "The item arrived damaged",
        "I want my money back",
        "The product is fake",
        "Extremely poor quality",
        "I am not satisfied",
        "Terrible experience",
        "Very uncomfortable to use",
        "It broke after first use",
        "I will never buy this again",
        "Really bad product"
    ],
    "Sentiment": [
        "Positive","Positive","Positive","Positive","Positive",
        "Positive","Positive","Positive","Positive","Positive",
        "Positive","Positive","Positive","Positive","Positive",
        "Positive","Positive","Positive","Positive","Positive",
        "Negative","Negative","Negative","Negative","Negative",
        "Negative","Negative","Negative","Negative","Negative",
        "Negative","Negative","Negative","Negative","Negative",
        "Negative","Negative","Negative","Negative","Negative"
    ]
}

df = pd.DataFrame(data)

df.head()
df.tail()

print(df.shape)

print(df.columns)

print(df.dtypes)

print(df.isnull().sum())

print(df["Sentiment"].value_counts())

df["Sentiment"].value_counts().plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()

    text = text.translate(str.maketrans("", "", string.punctuation))

    words = word_tokenize(text)

    words = [word for word in words if word not in stop_words]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

df["Clean_Review"] = df["Review"].apply(preprocess)

print(df[["Review", "Clean_Review"]].head())
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["Clean_Review"])

y = df["Sentiment"]

print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

new_reviews = [
    "This product is excellent and very useful",
    "I am unhappy with the quality",
    "Fast delivery and amazing service",
    "Waste of money and poor performance",
    "Good product at a reasonable price"
]

new_reviews = [preprocess(review) for review in new_reviews]

new_reviews_tfidf = vectorizer.transform(new_reviews)

predictions = model.predict(new_reviews_tfidf)

for review, prediction in zip(new_reviews, predictions):
    print("Review:", review)
    print("Predicted Sentiment:", prediction)
    print()
    print("Conclusion")
print("This project demonstrates a complete NLP pipeline using Python.")
print("The text data was preprocessed using tokenization, stop-word removal and lemmatization.")
print("TF-IDF was used to convert text into numerical features.")
print("A Naive Bayes classifier was trained to predict positive and negative sentiments.")
print("The model successfully classified new reviews based on their sentiment.")