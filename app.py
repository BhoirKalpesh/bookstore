import streamlit as st
import json
import spacy
import re

from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nlp = spacy.load('en_core_web_sm')

with open('books.json', 'r') as f:
    books = json.load(f)

book_titles = [book['book_title'] for book in books]
book_years = [datetime.strptime(book['published_date'], '%Y-%m-%d').year for book in books]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(book_titles)

def classify_message(message):
    message = message.lower()
    keywords = {
        "Order Status": ["order", "status", "track", "delivery"],
        "Review Submission": ["review", "feedback", "rate"],
        "Store Policy Enquiry": ["policy", "return", "exchange", "shipping", "refund"],
        "Book Search": ["search", "find", "looking for", "book title", "recommend"],
    }

    for category, keys in keywords.items():
        if any(key in message for key in keys):
            return category


    doc = nlp(message)
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            return "Book Search"

    return "Unknown"

def find_closest_title(user_input):
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_input_tfidf, tfidf_matrix)

    if cosine_similarities.size > 0:
        closest_index = np.argmax(cosine_similarities)
        closest_score = cosine_similarities[0][closest_index]

        if closest_score >= 0.1:  
            return book_titles[closest_index]
    return None

def find_closest_year(message):
    match = re.search(r'\b(19|20)\d{2}\b', message)
    if match:
        mentioned_year = int(match.group(0))
        closest_year = min(book_years, key=lambda x: abs(x - mentioned_year))
        closest_book = [book for book in books if datetime.strptime(book['published_date'], '%Y-%m-%d').year == closest_year]
        return closest_book[0]['book_title'] if closest_book else None
    return None


st.title('Bookstore Chatbot')

# User input
message = st.text_input("Enter your message:")
if st.button("Run"):
    if message:
        # Classify the message
        category = classify_message(message)
        st.write(f"**Category :** {category}")

        # Find the closest book title
        if category == "Book Search":
            closest_title = find_closest_title(message)
            st.write(f"**Matched Book Title:** {closest_title if closest_title else 'No relevant book was found.'}")

        # Check for publication year match
        closest_year_match = find_closest_year(message)
        if closest_year_match:
            st.write(f"**Closest Publication Year Book:** {closest_year_match}")
    else:
        st.write("Please enter a message to search for a book.")
