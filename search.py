import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from tqdm import tqdm, tqdm_notebook
import zipfile
import io

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

import streamlit as st



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess(raw_text):
    sentence = re.sub(r"Watch any video online with Open-SUBTITLES|Free Browser extension: osdb.link/ext", "", raw_text)
    sentence = re.sub(r"Please rate this subtitle at www.osdb.link/agwma|Help other users to choose the best subtitles", "", sentence)
    sentence = re.sub(r"[^a-zA-Z]", " ", sentence)
    sentence = sentence.lower()
    tokens = nltk.word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

def decode_method(binary_data):
    with io.BytesIO(binary_data) as f:
        with zipfile.ZipFile(f, 'r') as zip_file:
            subtitle_content = zip_file.read(zip_file.namelist()[0])
    return subtitle_content.decode('latin-1')

def load_data():
    conn = sqlite3.connect('C:\\Users\\hi\\Projects\\Live Projects\\Search Engine Intership Project\\eng_subtitles_database.db')
    dd = pd.read_sql_query("SELECT * FROM zipfiles", conn)
    return dd



def preprocess_and_vectorize(dd):
    dd['file_content'] = dd['content'].apply(decode_method)
    dd = dd.head(int(len(dd)*0.3))
    dd['subtitles'] = dd['file_content'].apply(preprocess)
    vectorizer = CountVectorizer(max_features=8000, min_df=2)
    bow_matrix = vectorizer.fit_transform(dd['subtitles'])
    return dd, vectorizer, bow_matrix


def main():
    st.title("Subtitle Search Engine")

    
    dd = load_data()

    
    dd, vectorizer, bow_matrix = preprocess_and_vectorize(dd)

    
    user_movie_dailogue = st.text_area("Enter any dialogue:", height=100)
    if st.button("Search"):
        if user_movie_dailogue:
            user_movie_dailogue_processed = preprocess(user_movie_dailogue)
            user_bow = vectorizer.transform([user_movie_dailogue_processed])

    
            cosine_similarities = cosine_similarity(user_bow, bow_matrix)
            top_5_movie_index = cosine_similarities.argsort(axis=None)[-5:][::-1]

            
            st.subheader("Top 5 Similar Dialogues:")
            for index in top_5_movie_index:
                most_similar_dialogues = dd.loc[index, 'chunked_data']
                st.write(f"Dialogue {index + 1}: {most_similar_dialogues}")

if __name__ == "__main__":
    main()
