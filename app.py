import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

# Download necessary NLTK resources if not already downloaded
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Download resources within the function, ensuring they're available when needed
    download_nltk_resources()
    
    # Preprocess the text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical characters
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load the trained model and vectorizers
model = joblib.load('nb_model.pkl')
vectorizer_title = joblib.load('tfidf_title.pkl')
vectorizer_text = joblib.load('tfidf_text.pkl')

# Streamlit app UI
st.title("Fake News Detection")
st.write("Enter a news article's title and text to check if it's real or fake.")

title_input = st.text_input("Title")
text_input = st.text_area("Text")

if st.button("Predict"):
    if not title_input or not text_input:
        st.warning("Please enter both title and text.")
    else:
        clean_title = preprocess_text(title_input)
        clean_text = preprocess_text(text_input)

        vec_title = vectorizer_title.transform([clean_title])
        vec_text = vectorizer_text.transform([clean_text])
        final_input = hstack([vec_title, vec_text])

        prediction = model.predict(final_input)[0]
        st.success(f"The news article is **{prediction.upper()}**.")

