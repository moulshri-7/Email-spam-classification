import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import copy

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()    # turns text to lower case
    text = nltk.word_tokenize(text)       # tokenizes text
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)     #removes special chars and only alpha-numeric words remain
    
    text = copy.deepcopy(y)
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:      #remove stopwords
            y.append(i)
    
    text = copy.deepcopy(y)
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))   #stemming of the text
    
    
    return " ".join(y)  #return list as a string

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")

input_mail = st.text_area("Enter the message: ")

if st.button('Predict'):


    #1. preprocess

    transformed_message = transform_text(input_mail)


    #2. vectorize

    vector_input = tfidf.transform([transformed_message])

    #3. predict

    result = model.predict(vector_input)[0]

    #4. display

    if result == 1:
        st.header("Spam")
    else:
        st.header("Ham")
