import os
import numpy as np
import pandas as pd
import json
import string
# import nlkt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    with open('mwalimu_sacco.json') as content:
        data = json.load(content)
except FileNotFoundError:
    raise FileNotFoundError(f'The file mwalimu_sacco.json was not found.')
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")
    print(f"Error on line {e.lineno}, column {e.colno}: {e.msg}")
    raise

tags = []
inputs = []
responses = {}
for intent in data['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['inputs']:
        inputs.append(lines)
        tags.append(intent['tag'])
df = pd.DataFrame({'inputs': inputs, 'tags': tags})
df['inputs'] = df['inputs'].apply(lambda wrd: ''.join([ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation]))

tokenizer = Tokenizer(num_words=50000) #20000
tokenizer.fit_on_texts(df['inputs'])
train = tokenizer.texts_to_sequences(df['inputs'])
x_train = pad_sequences(train)

# label encoding
le = LabelEncoder()
y_train = le.fit_transform(df['tags'])

# Define vocabulary
vocabulary = len(tokenizer.word_index)
output_length = len(le.classes_)

i = Input(shape=(x_train.shape[1],))
x = Embedding(vocabulary + 1, 10, input_length=x_train.shape[1])(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics = ['accuracy'])
# One-hot encode labels
y_train_one_hot = to_categorical(y_train, num_classes=output_length)

# Train model
try:
    train = model.fit(x_train, y_train_one_hot, epochs=200)
except Exception as e:
    raise ValueError(f"An unexpected error occurred during model training: {str(e)}")

# model visualization
plt.plot(train.history['accuracy'], label='Training Set Accuracy')
plt.plot(train.history['loss'], label='Training Set Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.show()

model.save("chatbot_model.h5")
# model.save("chatbot_model.keras")