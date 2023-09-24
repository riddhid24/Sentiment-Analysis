import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import re, string, unicodedata
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from keras.layers import LSTM, Dense, Bidirectional, Input
from keras.models import Model
import torch
import transformers
from sklearn.preprocessing import LabelBinarizer

# Load your dataset (replace with your dataset path)
reviews = pd.read_csv('IMDB Dataset.csv')

# Map sentiment labels to numeric values
reviews['sentiment_numeric'] = reviews['sentiment'].map({'positive': 1, 'negative': 0})

# Define text preprocessing function
def preprocess_review(review):
    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Convert text to lowercase
    review = review.lower()

    # Remove HTML tags using BeautifulSoup
    review = BeautifulSoup(review, 'html.parser').get_text()

    # Remove special characters, numbers, and punctuation
    review = re.sub(r'[^a-zA-Z]', ' ', review)

    # Tokenize the text
    words = word_tokenize(review)

    # Remove stopwords and perform lemmatization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]

    # Join the words back into a sentence
    preprocessed_review = ' '.join(words)

    return preprocessed_review

# Apply text preprocessing to the entire dataset
reviews['preprocessed_review'] = reviews['review'].apply(preprocess_review)

# Split the dataset into train, validation, and test sets
X = reviews['preprocessed_review']
y = reviews['sentiment_numeric']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Tokenize and pad sequences
max_len = 200  # Adjust the maximum sequence length as needed
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_val_sequences = tokenizer.texts_to_sequences(X_val)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len, padding='post')
X_val_padded = pad_sequences(X_val_sequences, maxlen=max_len, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len, padding='post')

# Define the model
embedding_dim = 100  # Adjust the embedding dimension as needed
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size

input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
lstm_layer = LSTM(64)(embedding_layer)
output_layer = Dense(1, activation='sigmoid')(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 5  # Adjust the number of epochs as needed
history = model.fit(
    X_train_padded,
    y_train,
    epochs=epochs,
    batch_size=64,  # Adjust batch size as needed
    validation_data=(X_val_padded, y_val)
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Make predictions
y_pred = model.predict(X_test_padded)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate additional evaluation metrics
accuracy = accuracy_score(y_test, y_pred_binary)
confusion = confusion_matrix(y_test, y_pred_binary)
classification_rep = classification_report(y_test, y_pred_binary)

# Print the results
print("Test Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(classification_rep)

# Visualize training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
