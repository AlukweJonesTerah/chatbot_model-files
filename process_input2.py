import random
import string
import os
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Initialize spaCy
nlp = spacy.load("en_core_web_md")

# Function to find similar queries
def find_similar_query(user_input, dataset_queries):
    user_input_embedding = nlp(user_input)
    similarities = [user_input_embedding.similarity(nlp(query)) for query in dataset_queries]

    most_similar_index = similarities.index(max(similarities))
    threshold = 0.7

    return dataset_queries[most_similar_index] if similarities[most_similar_index] >= threshold else None

# Function to find closest keyword
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

# Keyword Matching Function
def keyword_matching(user_input):
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

# Function to handle organization entities
def handle_organization_entities(intents, user_input):
    # Add logic to handle organization entities based on intents
    pass

# Function to analyze sentiment using TextBlob
def analyze_sentiment(user_input):
    text_blob_input = ' '.join([ltrs.lower() for ltrs in user_input.split() if ltrs not in string.punctuation])
    analysis = TextBlob(text_blob_input)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Function to analyze sentiment using VADER
def analyze_sentiment_vader(user_input):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(user_input)

    if sentiment_scores['compound'] >= 0.05:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Function to get a model confidence score
def get_model_confidence(user_input):
    # Add logic to get model confidence score
    pass

# Function to get a chatbot response
def get_chatbot_response(user_input):
    text_p = ' '.join([ltrs.lower() for ltrs in user_input.split() if ltrs not in string.punctuation])
    prediction_input = tokenizer.texts_to_sequences([text_p])
    prediction_input = pad_sequences(prediction_input, maxlen=input_shape)
    output = model.predict(prediction_input)

    max_confidence = np.max(output)
    response_tag = le.inverse_transform([output.argmax()])[0]

    confidence_threshold = 0.7

    if max_confidence < confidence_threshold:
        return "Chatbot: I'm not sure I understand. Can you please rephrase or provide more details?"

    if response_tag in tag_steps:
        steps = tag_steps[response_tag]
        if random.choice([True, False]):
            return f"Chatbot: {steps['list_format']}"
        else:
            return f"Chatbot: {steps['paragraph_format']}"

    return random.choice(responses.get(response_tag, responses["default"]))

# Function to log feedback to a file
def log_feedback_to_file(feedback):
    feedback_file_path = "user_feed.txt"
    mode = "a" if os.path.exists(feedback_file_path) else "w"
    with open(feedback_file_path, mode) as feedback_file:
        feedback_file.write(f"{feedback}\n")

# Function to collect user feedback
def collect_user_feedback():
    feedback = input("Chatbot: How would you rate this response? (1-5) (Good/Bad/Neutral): ")
    try:
        feedback = int(feedback)
        if 1 <= feedback <= 5:
            log_feedback_to_file(str(feedback))
            print("Chatbot: Thank you for your feedback!")
        else:
            print("Chatbot: Please provide a rating between 1 and 5.")
    except ValueError:
        print("Chatbot: Invalid input. Please provide a numeric rating.")

# Function to extract person name using spaCy
def extract_person_name(text):
    doc = nlp(text)
    person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return person_names[0] if person_names else None

# Function to handle person entities
def handle_person_entities(entities, context):
    for entity in entities:
        if entity.lower() == "person":
            person_name = extract_person_name(context["user_input"])

            if not person_name:
                person_name = input("You (Specify Person's Name): ")

            print(f"Chatbot: Hello {person_name}! How can I assist you today?")
            context["recognized_person"] = person_name
            break

# Function to extract entities using spaCy
def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

# User interaction loop
print("Welcome to the Chatbot!")
while True:
    try:
        user_input = input('You: ')

        conversation_history.append(user_input)
        context = ''.join(conversation_history[-3:])
        closest_keyword = find_closest_keyword(user_input, tag_keywords)
        similar_query = find_similar_query(user_input, df['inputs'])
        similarity_score = calculate_semantic_similarity(user_input, df['inputs'])
        print(f"Semantic Similarity Score: {similarity_score}")
        response = keyword_matching(user_input)
        handle_organization_entities(df['intents'], user_input)

        if similar_query:
            print(f"Chatbot: It seems like you meant '{similar_query}'. How can I assist you?")
        else:
            print(fallback_response())

        if closest_keyword:
            print(f"Chatbot: It seems like you are interested in '{closest_keyword}'. How can I assist you with {user_input}")
        else:
            sentiment = analyze_sentiment_vader(user_input)
            default_responses = {
                "greeting": "Hello! How can I help you today?",
                "default": "I'm sorry, I'm not sure how to respond to that. Could you please provide more details?"
            }

            chatbot_response = responses.get(response_tag,
                                             default_responses.get(response_tag, default_responses["default"]))
            if sentiment == 'positive':
                print("Chatbot: I'm glad you're feeling positive! ", get_chatbot_response(user_input))
                print(f"Chatbot: {chatbot_response}")
            elif sentiment == 'negative':
                print("Chatbot: I'm sorry to hear that. ", get_chatbot_response(user_input))
            else:
                model_confidence = get_model_confidence(user_input)
                confidence_threshold = 0.7

                if model_confidence >= confidence_threshold:
                    print(f"Chatbot: I'm {model_confidence * 100:.2f}% confident that the intent is {response_tag}.")
                else:
                    print("Chatbot: I'm not very confident in understanding your intent. Can you provide more information?")

                response_tag = le.inverse_transform([model.predict(pad_sequences(tokenizer.texts_to_sequences([user_input]), maxlen=input_shape)).argmax()])[0]
                print("Chatbot: ", get_chatbot_response(user_input))

                if response_tag == 'clarification':
                    print("Chatbot: Could you please provide more details or clarify your question?")
                    user_input_clarification = input('You (Clarification):')
                    conversation_history.append(user_input_clarification)
                    context += f'{user_input_clarification}'

                    sentiment_clarification = analyze_sentiment_vader(user_input_clarification)
                    if sentiment_clarification == 'positive':
                        print("Chatbot: Thanks for clarifying positively!", get_chatbot_response(user_input_clarification))
                    elif sentiment_clarification == 'negative':
                        print("Chatbot: I appreciate the clarification, even if it's negative.", get_chatbot_response(user_input_clarification))
                    else:
                        print("Chatbot: Thank for clarifying", get_chatbot_response(user_input_clarification))

                elif response_tag == "goodbye":
                    print("Chatbot: Goodbye! If you have more questions, feel free to ask.")
                    break

                entities = extract_entities(user_input)
                if "location" in entities:
                    print("Chatbot: I can help you with information about that location.", get_chatbot_response(user_input))
                elif "product" in entities:
                    print("Chatbot: Tell me more about the product you're interested in.", get_chatbot_response(user_input))

                handle_person_entities(entities, {'user_input': user_input})
                print("Chatbot: What else would like to know or discuss?")
        collect_user_feedback()
    except Exception as e:
        print(f"Error: {e}")
