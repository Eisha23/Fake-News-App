import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy
from scipy.sparse import hstack

# Download NLTK resources safely
def download_nltk_resources():
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Text preprocessing
def preprocess_text(text):
    download_nltk_resources()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load the saved model and vectorizers
model = joblib.load("nb_model.pkl")
vectorizer_title = joblib.load("tfidf_title.pkl")
vectorizer_text = joblib.load("tfidf_text.pkl")

# Streamlit UI
st.title("📰 Fake News Detection App")

st.markdown("Enter the **title** and **text** of a news article to predict if it's real or fake.")

title_input = st.text_input("News Title")
text_input = st.text_area("News Text")

if st.button("Predict"):
    if title_input.strip() == "" or text_input.strip() == "":
        st.warning("Please enter both title and text to get a prediction.")
    else:
        clean_title = preprocess_text(title_input)
        clean_text = preprocess_text(text_input)

        vec_title = vectorizer_title.transform([clean_title])
        vec_text = vectorizer_text.transform([clean_text])

        combined_features = hstack([vec_title, vec_text])

        prediction = model.predict(combined_features)[0]

        label = "REAL" if prediction == 1 else "FAKE"
        st.success(f"The news article is **{label}**.")

