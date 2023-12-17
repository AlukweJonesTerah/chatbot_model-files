import os
import zipfile

import nlp
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import string
import random

import wget as wget
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dropout,
    Dense,
    BatchNormalization,
    Flatten,
    Bidirectional,
    GRU
)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Attention
from textblob import TextBlob
import requests
from zipfile import ZipFile
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

nltk.download('vader_lexicon')
nltk.download('stopwords')

# Download GloVe embeddings (e.g., GloVe 100-dimensional embeddings)
glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
glove_zip_path = "glove.6B.zip"
glove_extract_path = "glove.6B.100d.txt"
# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Download GloVe file if not already downloaded
if not os.path.exists(glove_extract_path):
    try:
        response = requests.get(glove_url)
        response.raise_for_status()
        with open(glove_zip_path, "wb") as f:
            f.write(response.content)

        # Unzip GloVe file
        with zipfile.ZipFile(glove_zip_path, "r") as zip_ref:
            zip_ref.extractall()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading Glove file: {e}")
        # Handle the error appropriately, e.g., exit the progrom or use a default embedding
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Remove the zip file
        os.remove(glove_zip_path)

# Load data
try:
    with open('mwalimu_sacco.json') as content:
        data = json.load(content)
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")
    print(f"Error on line {e.lineno}, column {e.colno}: {e.msg}")
    raise

# Extract inputs and tags from JSON
tags = []
inputs = []
responses = {}
for intent in data["intents"]:
    responses[intent['tag']] = intent['responses']
    for lines in intent['inputs']:
        inputs.append(lines)
        tags.append(intent['tag'])
df = pd.DataFrame({"inputs": inputs, "tags": tags})

# Preprocess text data
df['inputs'] = df['inputs'].apply(
    lambda wrd: ' '.join([ltrs.lower() for ltrs in wrd.split() if ltrs not in string.punctuation]))

stop_words = set(stopwords.words('english'))
df['inputs'] = df['inputs'].apply(
    lambda wrd: ' '.join([ltrs.lower() for ltrs in wrd.split() if ltrs not in (string.punctuation, stop_words)])
)

# Tokenization and Padding
tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['inputs'])
train_sequences = tokenizer.texts_to_sequences(df['inputs'])
x_train = pad_sequences(train_sequences)

# Label Encoding
le = LabelEncoder()
y_train = le.fit_transform(df['tags'])

# Split the data into training  and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=seed
)

# Model
input_shape = x_train.shape[1]
vocabulary_size = len(tokenizer.word_index) + 1  # Adding 1 for the out-of-vocabulary token
output_length = len(set(df['tags']))

# Load GloVe embeddings into a dictionary
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create an embedding matrix
embedding_matrix = np.zeros((vocabulary_size, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Modify your model architecture to use the GloVe embeddings
i = Input(shape=(input_shape,))
x = Embedding(vocabulary_size, 100, weights=[embedding_matrix], trainable=False)(
    i)  # Use 100-dimensional GloVe embeddings
x = Bidirectional(LSTM(70, return_sequences=True))(x)
x = Bidirectional(LSTM(70, return_sequences=True))(x)
x = Bidirectional(GRU(50, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(68, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.3)(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)

# Adjust learning rate schedule
lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=100, decay_rate=0.9
)

# Use Adam optimizer with the adjusted learning rate
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),  # Adjust the learning rate
    metrics=["accuracy"],
)

# Define an early stopping callback
early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)

# Replace ExponentialDecay with ReduceLROnPlateau
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7
)

# Use ReduceLROnPlateau in fit method callbacks
history = model.fit(
    x_train,
    y_train,
    epochs=500,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping_callback, reduce_lr_callback],
    batch_size=32,
)

# Plot the training history
plt.plot(history.history["accuracy"], label="Training set accuracy")
plt.plot(history.history["loss"], label="Training set loss")
plt.plot(history.history["val_accuracy"], label="Validation set accuracy")
plt.plot(history.history["val_loss"], label="Validation set loss")
plt.legend()
plt.show()


# Function to get a chatbot response
def get_chatbot_response(user_input):
    text_p = ' '.join([ltrs.lower() for ltrs in user_input.split() if ltrs not in string.punctuation])
    prediction_input = tokenizer.texts_to_sequences([text_p])
    prediction_input = pad_sequences(prediction_input, maxlen=input_shape)
    output = model.predict(prediction_input)
    response_tag = le.inverse_transform([output.argmax()])[0]
    return random.choice(responses[response_tag])

# Sentiment analysis function using TextBlob (replace with your actual implementation if using a different library)

conversation_history = []
def analyze_sentiment(user_input):
    text_blob_input = ' '.join([ltrs.lower() for ltrs in user_input.split() if ltrs not in string.punctuation])
    analysis = TextBlob(text_blob_input)
    # Return 'positive', 'negative', or 'neutral' based on sentiment polarity
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

def analyze_sentiment_vader(user_input):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(user_input)

    if sentiment_scores['compound'] >= 0.05:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# User interaction loop
while True:
    try:
        user_input = input('You: ')

        # Add user input to conversation history
        conversation_history.append(user_input)

        # Retrieve context from conversation history
        context = ''.join(conversation_history)

        # Add sentiment analysis (you can use a library like TextBlob for simplicity)
        # sentiment = analyze_sentiment(user_input)
        sentiment = analyze_sentiment_vader(user_input)

        # Handle different sentiments
        if sentiment == 'positive':
            print("Chatbot: I'm glad you're feeling positive! ", get_chatbot_response(user_input))
        elif sentiment == 'negative':
            print("Chatbot: I'm sorry to hear that. ", get_chatbot_response(user_input))
        else:
            response_tag = le.inverse_transform([model.predict(pad_sequences(tokenizer.texts_to_sequences([user_input]), maxlen=input_shape)).argmax()])[0]
            print("Chatbot: ", get_chatbot_response(user_input))

        # Check for goodbye tag (make sure to define response_tag within the loop)
        if response_tag == "goodbye":
            break
    except Exception as e:
        print(f"Error: {e}")
