import os
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import string
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:

    print('File opened successfully')
except FileNotFoundError:
    raise FileNotFoundError(f'The file {data} was not found.')
except json.JSONDecodeError:
    raise ValueError(f'Error decoding JSON in file {data}')

# with open('mwalimu_sacco.json') as content:
#     print(content.read())

try:
    with open('mwalimu_sacco.json') as content:
        data = json.load(content)
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")
    print(f"Error on line {e.lineno}, column {e.colno}: {e.msg}")
    raise
print("Shape of data: ", len(data))

tags = []
inputs = []
responses = {}
for intent in data["intents"]:
    responses[intent['tag']] = intent['responses']
    for lines in intent['inputs']:
        inputs.append(lines)
        tags.append(intent['tag'])
df = pd.DataFrame({"inputs": inputs, "tags": tags})
df.head(10)
df['inputs'] = df['inputs'].apply(lambda wrd: ''.join([ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation]))
print(df.head(5))
tokenizer = Tokenizer(num_words=50000)  # 20000
tokenizer.fit_on_texts(df['inputs'])
train = tokenizer.texts_to_sequences(df['inputs'])
x_train = pad_sequences(train)

print("Shape of x_train: ", x_train)

# encoding the output
le = LabelEncoder()
y_train = le.fit_transform(df['tags'])
print("Shape of x_train: ", x_train)

input_shape = x_train.shape[1]
print(input_shape)
# Define vocabulary
vocabulary = len(tokenizer.word_index)
print("number of unique words: ", vocabulary)
output_length = le.classes_.shape[0]
print("output length: ", output_length)
# model
i = Input(shape=(input_shape))
x = Embedding(vocabulary + 1, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
try:
    train = model.fit(x_train, y_train, epochs=200)
except Exception as e:
    raise ValueError(f"Model training failed: {str(e)}")

# graph plotting
plt.plot(train.history['accuracy'], label='training set accuracy')
plt.plot(train.history['loss'], label='training set loss')
plt.legend()
plt.show()


