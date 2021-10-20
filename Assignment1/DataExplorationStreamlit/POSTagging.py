import requests
from bs4 import BeautifulSoup

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

from collections import Counter
import json, re, random
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from DataExploration import process_raw_data, clean_text
import spacy
import time
import streamlit as st

big_data_file = './reviewSelected100.json'

big_data = process_raw_data(big_data_file)

big_json = [json.loads(x) for x in big_data]

nlp_sm = spacy.load("en_core_web_sm")
nlp_trf = spacy.load("en_core_web_trf")

def analyze_pos(sampled_reviews):
    for review in sampled_reviews:
        cleaned_review = clean_text(review['text'])
        print(cleaned_review)
        tokenized_review = word_tokenize(cleaned_review)
        print(nltk.pos_tag(tokenized_review))  # NLTK POS taggers uses tag from the Penn Tree Bank tagset.
        print("")
        # To find another POS Tagging technique for comparison

def pos_spacy(sampled_reviews):
    # sm F1 for POS 0.97, trf F1 for POS 0.98
    pos_df = pd.DataFrame(columns=["token", "pos_1", "pos_2", "match"])
    for review in sampled_reviews:
        doc_sm = nlp_sm(review['text'])
        doc_trf = nlp_trf(review['text'])
        for token_sm, token_trf in zip(doc_sm, doc_trf):
            temp_dict = {
                "token": str(token_sm),
                "pos_1": token_sm.pos_,
                "pos_2": token_trf.pos_,
                "match": token_trf.pos_ == token_sm.pos_
            }
            pos_df = pos_df.append(temp_dict, ignore_index=True)
    return pos_df

def pos_spacy_coarse_fine(sampled_reviews):
    #sm F1 for POS 0.97, trf F1 for POS 0.98
    random_sents = []
    for review in sampled_reviews:
        doc_trf = nlp_trf(review['text'])
        random_sents.append(random.sample(sorted(doc_trf.sents),1)[0]) # Get 1 Random sentence from this review
    pos_df = pd.DataFrame(columns=["token", "pos_1", "pos_2", "match"])
    for sents in random_sents:
        for token_trf in sents:
            temp_dict = {
                "token":str(token_trf),
                "pos_1":token_trf.tag_,
                "pos_2":token_trf.pos_,
                "match":token_trf.pos_== token_trf.tag_
            }
            pos_df=pos_df.append(temp_dict, ignore_index=True)
    return pos_df

#random_reviews = random.sample(big_json,5)
# analyze_pos(random_reviews)
# df = pos_spacy(random_reviews)
#df = pos_spacy_coarse_fine(random_reviews)
#df.to_csv("POS_Tag.csv", index=None)
#df
df = pd.read_csv("POS_Tag.csv")

pos = 0
arr = [""]*(int(len(df)/2)+1)
for i, rows in df.iterrows():
    pos=pos%int(len(df)/2+1)
    arr[pos] = arr[pos]+str(i)+' & '+rows['token']+' & '+rows['pos_1']+' & '+rows['pos_2'] + '&'
    pos+=1

for i, row in enumerate(arr):
    arr[i] = row[:-1]+"\\\\"

for i in arr:
    st.write(i)
#tokenized_review = word_tokenize(random_reviews[0]['text'])
#doc_trf = nlp_trf(random_reviews[0]['text'])
#print(len(tokenized_review))
#len(doc_trf)
#for index, i in enumerate(doc_trf):
#    print(i, tokenized_review[index])
