import os
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dropout,
    Dense,
    BatchNormalization,
    Flatten,
)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.callbacks import EarlyStopping

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
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
df['inputs'] = df['inputs'].apply(lambda wrd: ''.join([ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation]))

# Tokenization and Padding
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(df['inputs'])
train = tokenizer.texts_to_sequences(df['inputs'])
x_train = pad_sequences(train)

# Label Encoding
le = LabelEncoder()
y_train = le.fit_transform(df['tags'])

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=seed
)

# Model
input_shape = x_train.shape[1]
vocabulary = len(tokenizer.word_index)
output_length = le.classes_.shape[0]

i = Input(shape=(input_shape,))
x = Embedding(vocabulary + 1, 20)(i)
x = LSTM(50, return_sequences=True)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(50, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)

# Learning Rate Schedule
lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=100, decay_rate=0.9
)

# Compile the model with adjusted learning rate using the schedule
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=lr_schedule),
    metrics=["accuracy"],
)

# Define an early stopping callback
early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# Fit the model with callbacks
history = model.fit(
    x_train,
    y_train,
    epochs=200,
    validation_data=(x_val, y_val),
    # callbacks=[early_stopping_callback],
)

# Plot the training history
plt.plot(history.history["accuracy"], label="Training set accuracy")
plt.plot(history.history["loss"], label="Training set loss")
plt.plot(history.history["val_accuracy"], label="Validation set accuracy")
plt.plot(history.history["val_loss"], label="Validation set loss")
plt.legend()
plt.show()
