import os
import random
import string
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from textblob import TextBlob
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Initialize conversation history
conversation_history = []

# Function to find a similar query in a dataset
def find_similar_query(user_input, dataset_queries):
    user_input_embedding = nlp(user_input)
    similarities = [user_input_embedding.similarity(nlp(query)) for query in dataset_queries]

    most_similar_index = similarities.index(max(similarities))
    threshold = 0.7

    return dataset_queries[most_similar_index] if similarities[most_similar_index] >= threshold else None

# Function to find the closest keyword
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

# Tag keywords
tag_keywords = {
    "loan_inquiry": ["loan", "apply", "requirements", "interest rates", "options", "types"],
    # Add keywords for other tags
}

# Function for keyword matching
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

# Function to calculate semantic similarity
def calculate_semantic_similarity(user_input, dataset_queries):
    user_input_embedding = nlp(user_input).vector
    dataset_embeddings = [nlp(query).vector for query in dataset_queries]
    similarities = cosine_similarity([user_input_embedding], dataset_embeddings)[0]
    return max(similarities)

# Function to handle organization entities
def handle_organization_entities(intents, user_input):
    # Implement organization-specific logic based on user input and intents
    pass

# Function to analyze sentiment using TextBlob
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

# Function to analyze sentiment using VADER Sentiment Analyzer
def analyze_sentiment_vader(user_input):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(user_input)

    if sentiment_scores['compound'] >= 0.05:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Function to get model confidence (placeholder)
def get_model_confidence(user_input):
    # Placeholder implementation; replace with actual logic
    return 0.8

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
        # Alternate between list and paragraph formats
        if random.choice([True, False]):  # Randomly choose between formats
            return f"Chatbot: {steps['list_format']}"
        else:
            return f"Chatbot: {steps['paragraph_format']}"

    # Use the response for the detected tag or a random response
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
            # Log the feedback (you can use a logging library or store in a database)
            log_feedback_to_file(str(feedback))  # Convert feedback to string before writing to file
            print("Chatbot: Thank you for your feedback!")
        else:
            print("Chatbot: Please provide a rating between 1 and 5.")
    except ValueError:
        print("Chatbot: Invalid input. Please provide a numeric rating.")

# Function to extract person name using spaCy
def extract_person_name(text):
    doc = nlp(text)
    # Extract person names using spaCy's named entity recognition
    person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    # Return the first person name if available, otherwise return None
    return person_names[0] if person_names else None

# Function to handle person entities
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

# Function to extract entities using spaCy
def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

# Function to generate a step response
def generate_step_response(steps):
    return "\n".join(steps)

# Tag steps
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

# Welcome message
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