import streamlit as st
from streamlit import session_state
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import json,re,random
import matplotlib.pyplot as plt
import spacy
import pandas as pd
import requests
from bs4 import BeautifulSoup

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

from collections import Counter
import json,re,random
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
import random
from DataExploration import process_raw_data
import json, re, random
import spacy
import DataExploration
from DataExploration import *
from POSTagging import *
from NounAdjPair import *
from IndicativeAdjectivePhrases import *
"""
# My first app
CE4045 NLP Assignment 1:
"""
st.set_page_config(page_title='My First App', page_icon=':smiley', layout="wide", initial_sidebar_state='expanded')

b1 = st.sidebar.button("1. Tokenization and Stemming", key="1")
b2 = st.sidebar.button("2. POS Tagging", key="2")
b3 = st.sidebar.button("3. Writing Style", key="3")
b4 = st.sidebar.button("4. Most frequent ⟨ Noun - Adjective ⟩ pairs for each rating", key="4")
b5 = st.sidebar.button("5. Extraction of Indicative Adjective Phrases", key="5")

ps = PorterStemmer()
lz = WordNetLemmatizer()
en_stopwords = stopwords.words('english')

# Load SpaCy models
# nlp_sm = spacy.load("en_core_web_sm")
# nlp_trf = spacy.load("en_core_web_trf")

# Need to run if stopwords not downloaded before.
### UNCOMMENT THIS WHEN SUBMITTING ###
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# Need to run if stopwords not downloaded before.

big_data_file = './reviewSelected100.json'

big_data = process_raw_data(big_data_file)

big_json = [json.loads(x) for x in big_data]

if b1:
    # Collecting all unique the business id
    business_id_list = list(
        {x['business_id'] for x in big_json})  # make it a set via set comprehension {}, then call tolist
    # Selecting a random business and their review
    chosen_id_1 = random.choice(business_id_list)
    chosen_business_1 = [x for x in big_json if x['business_id'] == chosen_id_1]
    chosen_id_2 = chosen_id_1
    while chosen_id_2 == chosen_id_1:
        chosen_id_2 = random.choice(business_id_list)
        chosen_business_2 = [x for x in big_json if x['business_id'] == chosen_id_2]
    DataExploration.analyze_business(chosen_business_1)
    DataExploration.analyze_business(chosen_business_2)
    # exec(open("DataExploration.py").read())

if b2:
    df = pd.read_csv("POS_Tag.csv")
    pos = 0
    arr = [""] * (int(len(df) / 2) + 1)
    for i, rows in df.iterrows():
        pos = pos % int(len(df) / 2 + 1)
        arr[pos] = arr[pos] + str(i) + ' & ' + rows['token'] + ' & ' + rows['pos_1'] + ' & ' + rows['pos_2'] + '&'
        pos += 1

    for i, row in enumerate(arr):
        arr[i] = row[:-1] + "\\\\"

    for i in arr:
        st.write(i)
if b3:
    st.write("Discussion points based on the formality of the way of writing, proper use of English sentence structure such as good grammar, proper pronouns, capitalization, and terms used in the posts.")
    st.subheader("Stack Overflow")

if b4:
    print("bye")
    business_dict = {}
    Restructure big_json to group by business & the stars
    for i in big_json:
        if business_dict.get(i['business_id']) == None:
            business_dict[i['business_id']] = {}
            business_dict[i['business_id']]['1'] = []
            business_dict[i['business_id']]['2'] = []
            business_dict[i['business_id']]['3'] = []
            business_dict[i['business_id']]['4'] = []
            business_dict[i['business_id']]['5'] = []
        if i['stars'] == 1:
            business_dict[i['business_id']]['1'].append(i)
        elif i['stars'] == 2:
            business_dict[i['business_id']]['2'].append(i)
        elif i['stars'] == 3:
            business_dict[i['business_id']]['3'].append(i)
        elif i['stars'] == 4:
            business_dict[i['business_id']]['4'].append(i)
        else:
            business_dict[i['business_id']]['5'].append(i)
    #Populate individual list by stars
    stars_1 = get_random_reviews(business_dict, 1, 50)
    stars_2 = get_random_reviews(business_dict, 2, 20)
    stars_3 = get_random_reviews(business_dict, 3, 20)
    stars_4 = get_random_reviews(business_dict, 4, 20)
    stars_5 = get_random_reviews(business_dict, 5, 20)

    stars_1_pt = generate_phrase_dict_tree(stars_1)
    stars_2_pt = generate_phrase_dict_tree(stars_2)
    stars_3_pt = generate_phrase_dict_tree(stars_3)
    stars_4_pt = generate_phrase_dict_tree(stars_4)
    stars_5_pt = generate_phrase_dict_tree(stars_5)

    st.write("1 Star")
    sorted_dict = {k: v for k, v in sorted(stars_1_pt.items(), key=lambda item: item[1], reverse=True)}.items()
    st.write(list(sorted_dict)[:10])
    st.write("2 Star")
    sorted_dict = {k: v for k, v in sorted(stars_2_pt.items(), key=lambda item: item[1], reverse=True)}.items()
    st.write(list(sorted_dict)[:10])
    st.write("3 Star")
    sorted_dict = {k: v for k, v in sorted(stars_3_pt.items(), key=lambda item: item[1], reverse=True)}.items()
    st.write(list(sorted_dict)[:10])
    st.write("4 Star")
    st.write = {k: v for k, v in sorted(stars_4_pt.items(), key=lambda item: item[1], reverse=True)}.items()
    st.write(list(sorted_dict)[:10])
    st.write("5 Star")
    sorted_dict = {k: v for k, v in sorted(stars_5_pt.items(), key=lambda item: item[1], reverse=True)}.items()
    st.write(list(sorted_dict)[:10])

    #Show dependency graph
    doc = nlp_trf("this is a very fat and orange cat")
    for token in doc:
        st.write(token.text, '|', token.dep_, '|', token.head.text, '|', token.head.pos_, '|',
                 [child for child in token.children])
    spacy.displacy.render(doc, style='dep')

if b5:
    print("hi")
#    Run IAP.py
    with open('business_adj_phrase.json', 'r') as fp:
        biz_phrases = json.load(fp)

    tf = pd.read_csv("tf.csv")
    tfl = pd.read_csv("tf-l.csv")

    tf = pd.read_csv("tfidf-ntn.csv")
    tfl = pd.read_csv("tfidf-ltn.csv")  # Export TF-IDF table

    indicative_phrase = {}
    with open('indicative_phrase-ltn.json', 'r') as fp:
        indicative_phrase = json.load(fp)

    indicative_phrase

    Show dependency graph
    doc = nlp_trf("The cat is fat and fluffy")
    for token in doc:
        st.write(token.text, '|', token.dep_, '|', token.head.text, '|', token.head.pos_, '|',
                 [child for child in token.children])
    spacy.displacy.render(doc, style='dep')

    s = tf.sum(axis=1)
    t = s.sort_values(ascending=False)
    t = t.reset_index()
    temph = t[0][:500]

    plt.bar(t.index[:500], temph)


state.sync()
