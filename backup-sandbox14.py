import logging
import os
import zipfile

import fuzz as fuzz
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
from fuzzywuzzy import fuzz # for string matching
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

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

# Define a dictionary to store steps for different tags
tag_steps = {
    "join_mwalimu_sacco": {
        "list_format": [
            "Fill out the membership application form.",
            "Submit the completed form along with the required documents.",
            "Wait for approval from Mwalimu Sacco.",
            "Upon approval, deposit the required membership fee.",
            "Congratulations! You are now a member of Mwalimu Sacco."
        ],
        "paragraph_format": (
            "To join Mwalimu Sacco, follow these steps:\n"
            "1. Fill out the membership application form.\n"
            "2. Submit the completed form along with the required documents.\n"
            "3. Wait for approval from Mwalimu Sacco.\n"
            "4. Upon approval, deposit the required membership fee.\n"
            "5. Congratulations! You are now a member of Mwalimu Sacco."
        )
    },
    "another_tag": {
        "list_format": [
            "Step 1: Do something.",
            "Step 2: Do something else.",
            "Step 3: Continue the process."
            # Add more steps as needed
        ],
        "paragraph_format": (
            "To perform another_tag, follow these steps:\n"
            "1. Do something.\n"
            "2. Do something else.\n"
            "3. Continue the process."
        )
    },
    # Add steps for other tags
}
"""
  should be in get_chatbot_response
# Check for specific tags that require step-by-step instructions
if response_tag == "join_mwalimu_sacco":
    # Provide a list of steps for joining Mwalimu Sacco
    steps = [
        "Fill out the membership application form.",
        "Submit the completed form along with the required documents.",
        "Wait for approval from Mwalimu Sacco.",
        "Upon approval, deposit the required membership fee.",
        "Congratulations! You are now a member of Mwalimu Sacco."
    ]
    return generate_step_response(steps) 
"""

# Function to generate a step response
def generate_step_response(steps):
    return "\n".join(steps)

# Function to get a chatbot response
def get_chatbot_response(user_input):
    # Preprocess user input
    text_p = ' '.join([ltrs.lower() for ltrs in user_input.split() if ltrs not in string.punctuation])
    # Tokenize and pad the input sequence
    prediction_input = tokenizer.texts_to_sequences([text_p])
    prediction_input = pad_sequences(prediction_input, maxlen=input_shape)
    # Get model predictions
    output = model.predict(prediction_input)
    # Get the maximum confidence and corresponding label
    max_confidence = np.max(output)
    response_tag = le.inverse_transform([output.argmax()])[0]

    # Set a confidence threshold for fallback
    confidence_threshold = 0.7

    if max_confidence < confidence_threshold:
        return "Chatbot: I'm not sure I understand. Can you please rephrase or provide more details?"

    # Check if the response_tag has associated steps
    if response_tag in tag_steps:
        steps = tag_steps[response_tag]
        return f"Chatbot: {generate_step_response(steps['list_format'])}"

    # Use the response for the detected tag or a random response
    return random.choice(responses.get(response_tag, responses["default"]))

def log_feedback_to_file(feedback):
    feedback_file_path = "user_feed.txt"
    mode = "a" if os.path.exists(feedback_file_path) else "w"
    with open(feedback_file_path, mode) as feedback_file:
        feedback_file.write(f"{feedback}\n")

def collect_user_feedback():
    feedback = input("Chatbot: How would you rate this response? (1-5) (Good/Bad/Neutral): ")
    try:
        feedback = int(feedback)
        if 1 <= feedback <= 5:
            # Log the feedback (you can use a logging library or store in a database)
            # log_user_feedback(feedback)
            log_feedback_to_file(str(feedback))  # Convert feedback to string before writing to file
            print("Chatbot: Thank you for your feedback!")
        else:
            print("Chatbot: Please provide a rating between 1 and 5.")
    except ValueError:
        print("Chatbot: Invalid input. Please provide a numeric rating.")

def log_user_feedback(feedback):
    # Add logic to log feedback, update the model, or perform other actions
    # For simplicity, let's print the feedback in this example
    print(f"Logged feedback: {feedback}")

def fallback_response():
    return "Chatbot: I'm sorry, I dont quite understander that. Can you please rephrase or provide more details"

# def generate_step_response(steps):
#     response = "Here are the steps you need to follow:\n"
#     for i, step in enumerate(steps, start=1):
#         response += f"{i}.{step}\n"
#     return response

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

# Load a pre-trained English model with word embeddings
nlp = spacy.load("en_core_web_md")
# def find_similar_query(user_input, dataset_queries):
#     # Calculate the similarity between the user input and each query in the dataset
#     user_input_embedding = nlp(user_input).vector
#     dataset_embeddings = [nlp(query).vector for query in dataset_queries]
#     similarities = cosine_similarity([user_input_embedding], dataset_embeddings)[0]
#
#     # Find the index of the most similar query
#     most_similar_index = similarities.argmax()
#
#     # set a threshold for considering a match
#     threshold = 0.7
#     if similarities[most_similar_index] >= threshold:
#         return dataset_queries[most_similar_index]
#     else:
#         return None

# find_similar query update
def find_similar_query(user_input, dataset_queries):
    user_input_embedding = nlp(user_input)
    similarities = [user_input_embedding.similarity(nlp(query)) for query in dataset_queries]

    most_similar_index = similarities.index(max(similarities))
    threshold = 0.7

    return dataset_queries[most_similar_index] if similarities[most_similar_index] >= threshold else None


def find_closest_keyword(user_input, keywords):
    user_input_lower = user_input.lower()
    closest_keyword = []
    max_similarity = 0.0

    for tag, tag_keywords in keywords.items():
        for keyword in tag_keywords:
            similarity = fuzz.ratio(user_input_lower, keyword.lower())
            if similarity > max_similarity:
                max_similarity = similarity
                closest_keyword = [keyword]
            elif similarity == max_similarity:
                closest_keyword.append(keyword)
    # Threshold for considering a match
    threshold = 70
    return closest_keyword if max_similarity >= threshold else None

tag_keywords = {
    "loan_inquiry": ["loan", "apply", "requirements", "interest rates", "options", "types"],
    # Add keywords for other tags
}

# keyword update
def keyword_matching(user_input):
    # Define a database of keywords and associated responses
    keyword_responses = {
        "loan": "We have various loan options available. How can I assist you with loans?",
        "greeting": "Hello! How can I help you today?",
        # Add more keyword responses as needed
    }

    closest_keywords = find_closest_keyword(user_input, tag_keywords)

    if closest_keywords:
        # Prioritize the tag with the highest match
        prioritized_tag = max(tag_keywords, key=lambda tag: fuzz.ratio(user_input.lower(), closest_keywords[0].lower()))
        return keyword_responses.get(prioritized_tag, "I'm not sure I understand. Could you please provide more details?")
    else:
        # If no keyword matches, return response or trigger further processing
        return "I'm not sure I understand. Could you please provide more details?"

# def keyword_matching(user_input):
#     # Define a database of keywords and associated responses
#     keyword_responses = {
#         "loan": "We have various loan options available. How can I assist you with loans?",
#         "greeting": "Hello! How can I help you today?",
#     }
#     for keyword, response in keyword_responses.items():
#         if keyword in user_input.lower():
#             return response
#     # If no keyword matches, return response or trigger further processing
#     return "I'm not sure I understand. Could you please provide more details?"

# def calculate_semantic_similarity(user_input, dataset):
#     #  spaCy's similarity score ranges from 0 to 1, where 1 indicates identical meaning
#     doc1 = nlp(user_input)
#     doc2 = nlp(dataset)
#     similarity = doc1.similarity(doc2)
#     return similarity

def calculate_semantic_similarity(user_input, dataset):
    # Use TF-IDF Vectorizer to convert text into numerical vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([user_input] + dataset)

    # Calculate cosine similarity between user input
    similarity_scores = cosine_similarity(vectors[0], vectors[1:])

    # The first score is the similarity with th user inout itself, so we skip it
    max_similarity = max(similarity_scores[0, 1:])

    return max_similarity

# Load spaCy for entity recognition
nlp = spacy.load("en_core_web_sm")
def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def  extract_person_name(text):
    doc = nlp(text)
    # Extract person names using spaCy's named entity recognition
    person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    # Return the first person name if available, otherwise return None
    return person_names[0] if person_names else None

def  extract_organization_name(text):
    doc = nlp(text)
    # Extract person names using spaCy's named entity recognition
    organization_name = [ent.text for ent in doc.ents if ent.label_ == "ORGANIZATION"]
    # Return the first person name if available, otherwise return None
    return organization_name[0] if organization_name else None

def extract_custom_entity_info(text):
    doc = nlp(text)
    # Extract person names using spaCy's named entity recognition
    custom_entity_info = [ent.text for ent in doc.ents if ent.label_ == "CUSTOM_ENTITY"]
    # Return the first person name if available, otherwise return None
    return custom_entity_info[0] if custom_entity_info else None

from dateutil import parser
from datetime import datetime, timedelta

def handle_date_entities(entities):
    for entity in entities:
        if entity.lower() == "date":
            user_input_date = input("You (Specify Date): ")

            try:
                parsed_date = parser.parse(user_input_date, fuzzy=True)
                current_datetime = datetime.now()

                # Check if the parsed date is in the future
                if parsed_date > current_datetime:
                    days_difference = (parsed_date - current_datetime).days

                    # Add logic for future date-related actions
                    if days_difference > 7:
                        print(f"Chatbot: I noted the date {parsed_date}. Let's plan for the future!")
                    else:
                        print(f"Chatbot: The date {parsed_date} is coming up soon. Get ready!")

                else:
                    print("Chatbot: That date seems to be in the past. Can you provide a future date?")
                    if current_datetime - parsed_date < timedelta(days=7):
                        print("Chatbot: It's still recent. Can you provide a new date or choose from options?")
                        # Add options for the user to choose from or ask for a new date
                    else:
                        print("Chatbot: It's quite far in the past. Please provide a more recent date.")
                        # Request a more recent date or provide options

            except ValueError:
                # Add logic for handling date parsing errors or asking for a valid date
                print("Chatbot: I couldn't understand the date. Please provide a valid date.")

            break

def handle_numeric_entities(entities):
    for entity in entities:
        if entity.isnumeric():
            numeric_value = int(entity)

            # Add logic for positive numeric values
            if numeric_value > 0:
                print(f"Chatbot: That's a positive number! Let's work with it.")
                #Todo: Add specific actions for positive numeric values

            # Add logic for negative numeric values
            elif numeric_value < 0:
                print(f"Chatbot: That's a negative number! What can we do with negative values?")
                #Todo: Add specific actions for negative numeric values

            # Add logic for the numeric value being zero
            else:
                print("Chatbot: The number is zero. What specific information or action would you like?")
                #Todo: Add specific actions for the numeric value being zero

            # Additional logic based on the numeric value
            if numeric_value % 2 == 0:
                print("Chatbot: It's an even number!")
                #Todo: Add specific actions for even numeric values
            else:
                print("Chatbot: It's an odd number!")
                #Todo: Add specific actions for odd numeric values

            #Todo: You can further customize the logic based on your specific use case

            break

def handle_custom_entities(entities, context):
    for entity in entities:
        if entity.lower() == "custom_entity":
            # Extract custom entity information from user input (you may replace this with your entity extraction logic)
            custom_entity_info = extract_custom_entity_info(context["user_input"])

            # If custom entity information is not extracted, prompt the user for details
            if not custom_entity_info:
                custom_entity_info = input("You (Specify Custom Entity Information): ")

            # Acknowledge the custom entity and add logic for custom entity-specific actions
            print(f"Chatbot: I recognize your custom entity. {custom_entity_info}. Let's proceed accordingly.")

            # Update context with the recognized custom entity information
            context["recognized_custom_entity"] = custom_entity_info

            # You can add additional logic or specific actions based on the recognized custom entity

            break

# Example usage:
handle_custom_entities(["custom_entity"], {"user_input": "Tell me about the custom entity"})

def handle_person_entities(entities, context):
    for entity in entities:
        if entity.lower() == "person":
            # Extract person's name using NLP library (replace this with your NLP library of choice)
            person_name = extract_person_name(context["user_input"])

            # If a name is not extracted, prompt the user to specify the person's name
            if not person_name:
                person_name = input("You (Specify Person's Name): ")

            # Greet the person and add logic for person-specific interactions
            print(f"Chatbot: Hello {person_name}! How can I assist you today?")

            # Update context with the recognized person's name
            context["recognized_person"] = person_name

            # You can add additional logic or specific actions based on the recognized person

            break

def handle_organization_entities(entities, context):
    for entity in entities:
        if entity.lower() == "organization":
            # Extract organization name from user input (you may replace this with your entity extraction logic)
            organization_name = extract_organization_name(context["user_input"])

            # If organization name is not extracted, prompt the user to specify the organization
            if not organization_name:
                organization_name = input("You (Specify Organization): ")

            # Provide information and add logic for organization-specific actions
            print(f"Chatbot: I have information about {organization_name}. What would you like to know?")

            # Update context with the recognized organization name
            context["recognized_organization"] = organization_name

            # You can add additional logic or specific actions based on the recognized organization

            break

# Example usage:
handle_organization_entities(["organization"], {"user_input": "Tell me about the organization"})

def get_model_confidence(user_input):
    # Your implementation to get model confidence
    pass

# User interaction loop
print("Welcome to the Chatbot!")
while True:
    try:
        user_input = input('You: ')

        # Add user input to conversation history
        conversation_history.append(user_input)

        # Retrieve context from conversation history (limiting to the last 3 interactions)
        context = ''.join(conversation_history[-3:])

        # Find the closest matching keyword for queries
        closest_keyword = find_closest_keyword(user_input, tag_keywords)
        similar_query = find_similar_query(user_input, df['inputs'])
        similarity_score = calculate_semantic_similarity(user_input, df['inputs'])
        print(f"Semantic Similarity Score: {similarity_score}")
        response = keyword_matching(user_input)
        handle_organization_entities(df['intents'], user_input)

        if similar_query:
            print(f"Chatbot: It seems like you meant '{similar_query}'. How can I assist you?")
        else:
            # Fallback response
            print(fallback_response())

        # Check if a close match for queries is found
        if closest_keyword:
            print(f"Chatbot: It seems like you are interested in '{closest_keyword}'. How can I assist you with {user_input}")
        else:
            # Add sentiment analysis (you can use a library like TextBlob for simplicity)
            # sentiment = analyze_sentiment(user_input)
            sentiment = analyze_sentiment_vader(user_input)

            default_responses = {
                "greeting": "Hello! How can I help you today?",
                "default": "I'm sorry, I'm not sure how to respond to that. Cloud you please provide more details?"
            }

            # Use the response for the detected tag or the default response
            chatbot_response = responses.get(response_tag,
                                             default_responses.get(response_tag, default_responses["default"]))
            if sentiment == 'positive':
                print("Chatbot: I'm glad you're feeling positive! ", get_chatbot_response(user_input))
                print(f"Chatbot: {chatbot_response}")
            elif sentiment == 'negative':
                print("Chatbot: I'm sorry to hear that. ", get_chatbot_response(user_input))
            else:
                # Get confidence score from your model
                model_confidence = get_model_confidence(user_input)

                # Compare with threshold
                confidence_threshold = 0.7
                if model_confidence >= confidence_threshold:
                    print(f"Chatbot: I'm {model_confidence * 100:.2f}% confident that the intent is {response_tag}.")
                else:
                    print(
                        "Chatbot: I'm not very confident in understanding your intent. Can you provide more information?")

                response_tag = le.inverse_transform([model.predict(pad_sequences(tokenizer.texts_to_sequences([user_input]), maxlen=input_shape)).argmax()])[0]
                print("Chatbot: ", get_chatbot_response(user_input))

                if response_tag == 'clarification':
                    print("Chatbot: Could you please provide more details or clarify your question?")
                    # Additional logic to handle clarification
                    user_input_clarification = input('You (Clarification):')
                    conversation_history.append(user_input_clarification)
                    # Additional processing for clarification (update the context with the clarification)
                    context += f'{user_input_clarification}'

                    # Reanalyze sentiment after clarification
                    sentiment_clarification = analyze_sentiment_vader(user_input_clarification)
                    if sentiment_clarification == 'positive':
                        print("Chatbot: Thanks for clarifying positively!", get_chatbot_response(user_input_clarification))
                    elif sentiment_clarification == 'negative':
                        print("Chatbot: I appreciate the clarification, even if it's negative.", get_chatbot_response(user_input_clarification))
                    else:
                        print("Chatbot: Thank for clarifying", get_chatbot_response(user_input_clarification))
                # Check for goodbye tag (make sure to define response_tag within the loop)
                elif response_tag == "goodbye":
                    print("Chatbot: Goodbye! If you have more questions, feel free to ask.")
                    break

                # Entity-specific
                entities = extract_entities(user_input)
                if "location" in entities:
                    print("Chatbot: I can help you with information about that location.", get_chatbot_response(user_input))
                elif "product" in entities:
                    print("Chatbot: Tell me more about the product you're interested in.", get_chatbot_response(user_input))

                handle_person_entities(entities, {'user_input': user_input})
                # Provide instruction for the next user input
                print("Chatbot: What else would like to know or discuss?")
        collect_user_feedback()
    except Exception as e:
        print(f"Error: {e}")

logging.basicConfig(filename='chatbot.log', level=logging.INFO, format='%(asctime)s = %(levelname)s - %(message)s')
def log_interaction(user_input, chatbot_response):
    logging.info(f"User Input: {user_input}")
    logging.info(f"Chatbot Response: {chatbot_response}")

log_interaction(user_input, chatbot_response)