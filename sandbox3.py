import os
import numpy as np
import pandas as pd
import json
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_model(vocabulary, output_length, input_length):
    i = Input(shape=(input_length,))
    x = Embedding(vocabulary + 1, 10, input_length=input_length)(i)
    x = LSTM(128, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)  # Adding dropout for regularization
    x = Dense(output_length, activation="softmax")(x)
    model = Model(i, x)
    return model

def step_decay(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 20
    lr = initial_lr * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return lr

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

tokenizer = Tokenizer(num_words=350000)
tokenizer.fit_on_texts(df['inputs'])
train = tokenizer.texts_to_sequences(df['inputs'])
x_train = pad_sequences(train)

# label encoding
le = LabelEncoder()
y_train = le.fit_transform(df['tags'])

# Define vocabulary
vocabulary = len(tokenizer.word_index)
output_length = len(le.classes_)

# Split data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Build and compile the model
model = build_model(vocabulary, output_length, x_train.shape[1])
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])

# One-hot encode labels for training and test sets
y_train_one_hot = to_categorical(y_train, num_classes=output_length)
y_test_one_hot = to_categorical(y_test, num_classes=output_length)

# Learning rate scheduler
lr_schedule = LearningRateScheduler(step_decay)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# Train model with learning rate scheduler
try:
    train = model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot),
                      epochs=70, callbacks=[lr_schedule])
except Exception as e:
    raise ValueError(f"An unexpected error occurred during model training: {str(e)}")

# Model visualization
plt.plot(train.history['accuracy'], label='Training Set Accuracy')
plt.plot(train.history['loss'], label='Training Set Loss')
plt.plot(train.history['val_accuracy'], label='Validation Set Accuracy')

plt.plot(train.history['val_loss'], label='Validation Set Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.show()

# Save the trained model with the standard extension .h5
model.save("chatbot_model.h5")