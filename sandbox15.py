import os
import zipfile
import numpy as np
import pandas as pd
import json
import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dropout,
    Dense,
    BatchNormalization,
    Flatten,
    Bidirectional,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from textblob import TextBlob
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')
nltk.download('stopwords')

def load_data():
    try:
        with open('mwalimu_sacco.json') as content:
            data = json.load(content)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print(f"Error on line {e.lineno}, column {e.colno}: {e.msg}")
        raise

    tags = []
    inputs = []
    responses = {}
    for intent in data["intents"]:
        responses[intent['tag']] = intent['responses']
        for lines in intent['inputs']:
            inputs.append(lines)
            tags.append(intent['tag'])
    df = pd.DataFrame({"inputs": inputs, "tags": tags})
    return df, responses

def preprocess_text(text, stop_words):
    return ' '.join([ltrs.lower() for ltrs in text.split() if ltrs not in string.punctuation and ltrs not in stop_words])

def tokenize_and_pad(texts, max_words=50000, max_len=None):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return tokenizer, padded_sequences

def load_glove_embeddings(file_path='glove.6B.100d.txt'):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def create_model(input_shape, vocabulary_size, embedding_matrix, output_length):
    i = Input(shape=(input_shape,))
    x = Embedding(vocabulary_size, 100, weights=[embedding_matrix], trainable=False)(i)
    x = Bidirectional(LSTM(70, return_sequences=True))(x)
    x = Bidirectional(LSTM(70, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(68, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(output_length, activation="softmax")(x)
    model = Model(i, x)
    return model

def main():
    # Load data
    df, responses = load_data()

    # Preprocess text data
    stop_words = set(stopwords.words('english'))
    df['inputs'] = df['inputs'].apply(lambda wrd: preprocess_text(wrd, stop_words))

    # Tokenization and Padding
    tokenizer, x_train = tokenize_and_pad(df['inputs'], max_words=50000, max_len=None)

    # Label Encoding
    le = LabelEncoder()
    y_train = le.fit_transform(df['tags'])

    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Load GloVe embeddings into a dictionary
    embeddings_index = load_glove_embeddings()

    # Create an embedding matrix
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Model
    input_shape = x_train.shape[1]
    vocabulary_size = len(tokenizer.word_index) + 1  # Adding 1 for the out-of-vocabulary token
    output_length = len(set(df['tags']))

    model = create_model(input_shape, vocabulary_size, embedding_matrix, output_length)

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


if __name__ == "__main__":
    main()
