import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, plot_roc_curve
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


import streamlit as st



st.sidebar.title('Team 35 Sarcasm Redditector')

page = st.sidebar.radio(
    "Select Model",
    ("Both Models","TF-IDF", "LSTM")
)

# Load tf-idf model
tf_idf = pickle.load(open("tf_idf.pickle", "rb"))
logit = pickle.load(open("logit.pickle", "rb"))

def predict_tdidf(s):
    testing = tf_idf.transform([s])
    pred = logit.predict_proba(testing)
    st.write('**Probability of sarcasm:**', round(pred[0][1], 4))
    if pred[0][1] >= 0.5: # not too sure which is which though
        return "It's a sarcastic comment 🤡" 
    else:
        return "It's not sarcastic comment 😎"

# Load LSTM Model
tokenizer_obj = pickle.load(open("tokenizer.pickle", "rb"))
lstm_model = keras.models.load_model("finalLSTM")

@tf.autograph.experimental.do_not_convert
def predict_lstm(s):
    text_lines = pd.DataFrame({s:[]})
    print(text_lines)
    test_sequences = tokenizer_obj.texts_to_sequences(text_lines)
    test_review_pad = pad_sequences(test_sequences, maxlen=25, padding='post')
    pred = lstm_model.predict(test_review_pad)
    st.write('Probability of sarcasm:', round(pred[0][0], 4))
    if pred[0][0]>=0.5: 
        return "It's a sarcastic comment 🤡" 
    else: 
        return "It's not sarcastic comment 😎"

if page == 'Both Models' :
    selected_text = st.selectbox('Select one statement', ['yeah right', 'dogs'])
    col1, col2 = st.columns(2)
    with col1:
        st.write('## TF-IDF Model Prediction')
        st.write(predict_tdidf(selected_text))

    with col2:
        st.write('## LSTM Model Prediction')
        st.write(predict_lstm(selected_text))

if page == 'TF-IDF' :
    st.title('TF-IDF Model Prediction')

    text = st.text_input("Input your text")
    st.write(predict_tdidf(text))

if page == 'LSTM' :
    st.title('LSTM Model Prediction')
    
    text = st.text_input("Input your text")
    st.write(predict_lstm(text))
    


