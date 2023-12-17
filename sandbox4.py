import os
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    with open('mwalimu_sacco.json') as content:
        data = json.load(content)
    print('File opened successfully')
except FileNotFoundError:
    raise FileNotFoundError("The file 'mwalimu_sacco.json' was not found.")
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")
    print(f"Error on line {e.lineno}, column {e.colno}: {e.msg}")
    raise

# Process data
tags = []
inputs = []
responses = {}

for intent in data["intents"]:
    responses[intent['tag']] = intent['responses']
    for lines in intent['inputs']:
        inputs.append(lines)
        tags.append(intent['tag'])

df = pd.DataFrame({"inputs": inputs, "tags": tags})
df['inputs'] = df['inputs'].apply(lambda wrd: ''.join([ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation]))

# Tokenization
tokenizer = Tokenizer(num_words=350000)  # todo: change from 50,000
tokenizer.fit_on_texts(df['inputs'])
train = tokenizer.texts_to_sequences(df['inputs'])
x_train = pad_sequences(train)

# Encoding the output
le = LabelEncoder()
y_train = le.fit_transform(df['tags'])

# Define vocabulary
vocabulary = len(tokenizer.word_index)
output_length = len(le.classes_)

# One-hot encode labels
y_train_one_hot = to_categorical(y_train, num_classes=output_length)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train_one_hot, test_size=0.2, random_state=42)

# Define the model
i = Sequential()
i = Input(shape=(x_train.shape[1],))
x = Embedding(vocabulary + 1, 20, input_length=x_train.shape[1])(i)
x = LSTM(256, dropout=0.01, recurrent_dropout=0.01, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=['accuracy']) # , optimizer='adam'

# Train the model using the training set
try:
    train_history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=58)
except Exception as e:
    raise ValueError(f"An unexpected error occurred during model training: {str(e)}")

# Model visualization
plt.plot(train_history.history['accuracy'], label='Training Set Accuracy')
plt.plot(train_history.history['val_accuracy'], label='Validation Set Accuracy')
plt.plot(train_history.history['loss'], label='Training Set Loss')
plt.plot(train_history.history['val_loss'], label='Validation Set Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.show()

model.save("chatbot_model.h5")