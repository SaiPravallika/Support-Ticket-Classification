# -*- coding: utf-8 -*-
"""Preproecessing.ipynb


## Data pre-processing and privacy cleaning
1. Redact PII
2. Convert all text to lower case
3. Remove numbers
4. Remove puntuations
5. Remove blank spaces
6. Remove stop words
7. Remove duplicate rows
8. Filter English language support tickets
9. Filtering Long and short descriptions
10. Removing typos
"""

import re, unicodedata
import pandas as pd
import spacy
from textblob import TextBlob
import numpy as np
import nltk as nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import sys

"""Downloading Spacy English model"""
# !{sys.executable} -m spacy download en

allstp=np.array(stopwords.words('english'))

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s]{7,}\d)')
EMPID_RE = re.compile(r'\bEMP\d{4,6}\b', re.IGNORECASE)
MFA_RE = re.compile(r'\b\d{6}\b')
COORD_RE = re.compile(r'\b-?\d{1,3}\.\d{3,},-?\d{1,3}\.\d{3,}\b')
REF_RE = re.compile(r'\bRef:\s*\d+-\d+\b', re.IGNORECASE)

def _redact(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = EMAIL_RE.sub('[email]', text)
    text = EMPID_RE.sub('[id]', text)
    text = MFA_RE.sub('[code]', text)
    text = COORD_RE.sub('[god]', text)
    text = PHONE_RE.sub('[tel_num]', text)
    text = REF_RE.sub('[ref]]', text)
    return text


def normalize_text(text):
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s.,?!]', '', text)
    return text

"""Memory Constraint """
def correct_typos(text):
    if pd.isna(text):
        return text
    blob = TextBlob(text)
    return str(blob.correct())

"""Memory Constraint"""
def lemmetization(text):
  nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
  doc = nlp(text)
  lemmas = [token.lemma_ for token in doc]
  return " ".join(lemmas)


def remove_stopwords(text):
    if pd.isna(text):
        return text
    #print(allstp)
    #allstp=np.array(stopwords.words('english'))
    #Creating an additional of stopwords that  we see as irrelevant to the modelling inputs
    new_words=np.array(['yes','hi', 'receive','hello','sir','madam', 'best','morning','evening','afternoon' 'regards','thanks','from','greeting', 'forward','reply','will','please','see','help','able'])
    stopwords=np.concatenate([allstp,new_words]) #Concatenating nltk list and our list of stopwords

    words = [w for w in text.split() if w not in stopwords]
    return " ".join(words)


def clean_dataframe(df: pd.DataFrame,
                    max_title_len:int=180,
                    max_desc_len:int=1200,
                    min_len:int=10) -> pd.DataFrame:
    cols = ["title","description","answer","type","category","priority","language","tag"]
    df = df[[c for c in cols if c in df.columns]].copy()

    ##Apply Redact function
    for col in ["title","description","answer"]:
      if col in df.columns:
        df[col] = df[col].astype(str).apply(_redact)
        ## Apply Normalize function
        df[col] = df[col].astype(str).apply(normalize_text)
        ## Apply typos correct function
        #df[col] = df[col].astype(str).apply(correct_typos)
        ##Apply lemmetization
        #df[col] = df[col].apply(lemmetization)
        ## Apply stopwords removal
        df[col] = df[col].apply(remove_stopwords)

    if "title" in df.columns:
        df["title"] = df["title"].str.slice(0, max_title_len)
    if "description" in df.columns:
        df["description"] = df["description"].str.slice(0, max_desc_len)
    def too_short(row):
        t = (row.get("title","") or "")
        d = (row.get("description","") or "")
        return (len(t) < min_len) or (len(d) < min_len)
    df = df[~df.apply(too_short, axis=1)]
    dedup_key = (df["title"].fillna("") + "||" + df["description"].fillna("")).astype(str)
    df = df.loc[~dedup_key.duplicated(keep="first")].reset_index(drop=True)
    df = df.loc[df['language']=='en']

    return df