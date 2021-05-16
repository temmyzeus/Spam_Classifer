#!.env/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:46:18 2020

@author: Awoyele Temiloluwa Micheal
"""

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from string import punctuation
import re
import pickle

data = pd.read_csv("datasets/spam_ham_dataset.csv", encoding="Windows-1252")

# for every line where feature "Unnamed: 2" is not_null, we add Feature "Unnamed: 2" to "Text" feature
data["v2"][data["Unnamed: 2"].notna()] = data["v2"][data["Unnamed: 2"].notna()] + \
    data["Unnamed: 2"][data["Unnamed: 2"].notna()]
# for every line where feature "Unnamed: 3" is not_null, we add Feature "Unnamed: 3" to "Text" feature
data["v2"][data["Unnamed: 3"].notna()] = data["v2"][data["Unnamed: 3"].notna()] + \
    data["Unnamed: 3"][data["Unnamed: 3"].notna()]
# for every line where feature "Unnamed: 4" is not_null, we add Feature "Unnamed: 4" to "Text" feature
data["v2"][data["Unnamed: 4"].notna()] = data["v2"][data["Unnamed: 4"].notna()] + \
    data["Unnamed: 4"][data["Unnamed: 4"].notna()]
# drop feature "A","B", and "C"
data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
data.columns = ["label", "text"]  # rename columns

tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S=+')
lemma = WordNetLemmatizer()
tfidf = TfidfVectorizer()
rf_clf = RandomForestClassifier()


def remove_html(text):
    text = re.sub(r"http://.+\.com", "", text)
    text = re.sub(r"http://\S+", "", text)
    text = re.sub(r"http://[A-Za-z0-9./?= *&:-]+", "", text)
    text = re.sub(r"http.+\.com", "", text)
    text = re.sub(r"http//[A-Za-z0-9./?= *&:-]+", "", text)
    text = re.sub(r"www[.A-Za-z0-9/+-]+", "", text)
    return text


def remove_numbers(text):
    text = re.sub(r"[å£A-Za-z0-9]*\d+[A-Za-z0-9]*", "", text)
    return text


def remove_fancy_text(word):
    if word.isascii() == False:
        return ""
    else:
        return word


def remove_mails(text):
    text = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+(\.[a-zA-Z]{2,5})", "", text)
    return text


def remove_stopwords(text):
    return [word for word in text if word not in stopwords.words("english")]


def remove_puncts(text):
    return [word for word in text if word not in punctuation]


def remove_standalone_letters(text_tokens):
    return [word for word in text_tokens if len(word) >= 2]


def cleanText(text):
    text = str(text).lower()
    text = remove_html(text)
    text = remove_numbers(text)
    text = " ".join([remove_fancy_text(word) for word in text.split()])
    text = remove_mails(text)
    text_tokens = tokenizer.tokenize(text)
    text_tokens = remove_stopwords(text_tokens)
    text_tokens = remove_puncts(text_tokens)
    text_tokens = remove_standalone_letters(text_tokens)
    text = " ".join([lemma.lemmatize(word) for word in text_tokens])
    return text


label_map = {"ham": 0,
             "spam": 1}
data["label"] = data["label"].map(label_map)

data["text"] = data["text"].apply(cleanText)

X = tfidf.fit_transform(data["text"])
pickle.dump(tfidf, open("models/Tfidf_Convert.pkl", "wb"))
y = data["label"]

rf_clf.fit(X, y)
pickle.dump(rf_clf, open("models/model.pkl", "wb"))
