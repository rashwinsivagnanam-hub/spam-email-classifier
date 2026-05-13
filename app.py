import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Text cleaning function
def clean_text(text):

    text = text.lower()

    # Remove punctuation
    text = ''.join(
        char for char in text
        if char not in string.punctuation
    )

    # Remove stopwords
    words = text.split()

    words = [
        word for word in words
        if word not in stopwords.words('english')
    ]

    return ' '.join(words)

# Streamlit UI
st.title("Spam/Ham Email Classifier")

st.write("Enter a message below.")

message = st.text_area("Message")

if st.button("Predict"):

    cleaned_message = clean_text(message)

    vector_input = vectorizer.transform([cleaned_message])

    prediction = model.predict(vector_input)

    if prediction[0] == 1:
        st.error("SPAM MESSAGE")
    else:
        st.success("HAM MESSAGE")