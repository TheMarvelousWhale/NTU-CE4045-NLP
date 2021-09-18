import streamlit as st
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

st.set_page_config(page_title='My First App', page_icon=':smiley',
                   layout="wide", initial_sidebar_state='expanded')

b1 = st.sidebar.button("1. Tokenization and Stemming", key="1")
b2 = st.sidebar.button("2. POS Tagging", key="2")
b3 = st.sidebar.button("3. Writing Style", key="3")
b4 = st.sidebar.button("4. Most frequent ⟨ Noun - Adjective ⟩ pairs for each rating", key="4")
b5 = st.sidebar.button("5. Extraction of Indicative Adjective Phrases", key="5")
"""
# My first app
CE4045 NLP Assignment 1:
"""
ps = PorterStemmer()
lz = WordNetLemmatizer()
en_stopwords = stopwords.words('english')

# Load SpaCy models
nlp_sm = spacy.load("en_core_web_sm")
nlp_trf = spacy.load("en_core_web_trf")

# Need to run if stopwords not downloaded before.
### UNCOMMENT THIS WHEN SUBMITTING ###
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# Need to run if stopwords not downloaded before.

big_data_file = './reviewSelected100.json'

big_data = process_raw_data(big_data_file)

big_json = [json.loads(x) for x in big_data]

#Collecting all unique the business id
business_id_list = list({x['business_id'] for x in big_json})  #make it a set via set comprehension {}, then call tolist

#Selecting a random business and their review
chosen_id_1 = random.choice(business_id_list)
chosen_business_1 = [x for x in big_json if x['business_id'] == chosen_id_1]

chosen_id_2 = chosen_id_1
while chosen_id_2 == chosen_id_1:
    chosen_id_2 = random.choice(business_id_list)
    chosen_business_2 = [x for x in big_json if x['business_id'] == chosen_id_2]

pt_random_reviews = random.sample(big_json, 5)

stars_1 = []
stars_2 = []
stars_3 = []
stars_4 = []
stars_5 = []
for i in big_json:
    if i['stars'] == 1:
        stars_1.append(i)
    elif i['stars'] == 2:
        stars_2.append(i)
    elif i['stars'] == 3:
        stars_3.append(i)
    elif i['stars'] == 4:
        stars_4.append(i)
    else:
        stars_5.append(i)

nap_random_reviews = random.sample(stars_1, 5)

if b1:
        DataExploration.analyze_business(chosen_business_1)
        DataExploration.analyze_business(chosen_business_2)

if b2:
    pos_df = pos_spacy(pt_random_reviews)
    pos_df

if b3:
    st.write("Discussion points based on the formality of the way of writing, proper use of English sentence structure such as good grammar, proper pronouns, capitalization, and terms used in the posts.")
    st.subheader("Stack Overflow")

if b4:
    phrase_dict_1 = generate_phrase_dict(random_reviews)
    phrase_dict_1

if b5:
    df = pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    })

