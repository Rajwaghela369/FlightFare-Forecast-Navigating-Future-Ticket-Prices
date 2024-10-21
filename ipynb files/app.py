import streamlit as st
import pickle
import numpy as np
import re
import string
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

summarizer = pipeline('summarization')
with open('SVC.pkl', 'rb') as model_file:
    svc = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

labels = {0: 'Business', 1: 'Entertainment', 2: 'Health', 3: 'Science', 4: 'Sports', 5: 'Technology', 6: 'World'}

st.title('News Classification and Summarization', False)
st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:grey;" /> """, unsafe_allow_html=True)
url = st.text_input(label='Provide News URL:')

def process_text(text):

    text = re.sub(r'https?://\S+|ftp://\S+|www\.\S+', '', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '', text)
    text = re.sub(r'\b(?:\+?\d{1,3}[-.])?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = ''.join(char for char in text if char in string.ascii_letters or char.isspace())
    
    return text

    
if url!='':
    article = Article(url=url)
    article.download()
    article.parse()
    cleaned_text = process_text(article.text)
    content_vec = vectorizer.transform([cleaned_text]).toarray()
    pred = svc.predict(content_vec)
    st.subheader(f'Category: {labels[int(pred)]}')
    st.title(article.title)
    st.subheader(article.authors[0])
    date = str(article.publish_date)
    date = date.strip()[0:10]
    st.subheader(date)
    st.image(article.top_image)
    summary = summarizer(cleaned_text, max_length=700, min_length=300, do_sample=False)
    tab1, tab2 = st.tabs(['Summary', 'News'])
    with tab1:
        st.write(summary[0]['summary_text'])
    with tab2:
        st.write(article.text)