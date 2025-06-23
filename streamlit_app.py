import streamlit as st
import pickle
import nltk
import string

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()

try:
    stopwords.words('english')
    var = string.punctuation
    nltk.word_tokenize('')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')  # Added punkt_tab download


def transformed_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

st.title("Spam Detection")


input_sms = st.text_area("",placeholder="Enter  your message here...")

if st.button("Predict"):
    # preprocessing
    transformed_sms = transformed_text(input_sms)
    # vectorize
    vector_input = tfidf.transform([transformed_sms])
    # predict
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("This is a spam message")
    else:
        st.header("This is not a spam message")
