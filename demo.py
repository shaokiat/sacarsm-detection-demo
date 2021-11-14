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

import streamlit as st

add_selectbox = st.sidebar.radio(
    "Select Model",
    ("TF-IDF", "LSTM")
)

train_df = pd.read_csv("train-balanced-sarcasm.csv")
train_df.head()

train_df.dropna(subset=['comment'], inplace=True)

train_texts, valid_texts, y_train, y_valid = train_test_split(train_df['comment'], train_df['label'], random_state=17)

train_df.loc[train_df['label'] == 1, 'comment'].str.len().apply(np.log1p).hist(label='sarcastic', alpha=.5)
train_df.loc[train_df['label'] == 0, 'comment'].str.len().apply(np.log1p).hist(label='normal', alpha=.5)
plt.legend();

tf_idf = TfidfVectorizer(ngram_range=(1, 1), max_features=500, min_df=2)
logit = sklearn.linear_model.LogisticRegression(C=1, n_jobs=4, solver='lbfgs', verbose=1)

X_train_texts = tf_idf.fit_transform(train_texts)
X_valid_texts = tf_idf.transform(valid_texts)

logit.fit(X_train_texts, y_train)

valid_pred = logit.predict(X_valid_texts)
valid_pred_proba = logit.predict_proba(X_valid_texts)

acc = accuracy_score(y_valid, valid_pred)

st.write(acc)

def predict_sarcasm(s):
    testing = tf_idf.transform([s])
    pred = logit.predict_proba(testing)
    print(pred)
    if pred[0][1] >= 0.5: # not too sure which is which though
        return "It's a sarcasm"
    else:
        return "Not a sarcasm"

text = st.text_input("Input your text")
st.write(predict_sarcasm(text))