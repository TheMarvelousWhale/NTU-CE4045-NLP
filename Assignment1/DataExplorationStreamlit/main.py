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
"""
# My first app
Here's our first attempt at using data to create a table:
"""
ps = PorterStemmer()
lz = WordNetLemmatizer()
en_stopwords = stopwords.words('english')

# Load SpaCy models
nlp_sm = spacy.load("en_core_web_sm")
nlp_trf = spacy.load("en_core_web_trf")

# Need to run if stopwords not downloaded before.
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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

with st.expander("Task 3.2.1: Tokenization and Stemming"):
        DataExploration.analyze_business(chosen_business_1)
        DataExploration.analyze_business(chosen_business_2)

with st.expander("Task 3.2.2: POS Tagging"):
    df = pd.DataFrame({
        'first column': [1, 2, 3, 4],
         'second column': [10, 20, 30, 40]
    })
with st.expander("Task 3.2.3: Writing Style"):
    st.write("Discussion points based on the formality of the way of writing, proper use of English sentence structure such as good grammar, proper pronouns, capitalization, and terms used in the posts.")
    st.subheader("Stack Overflow")

with st.expander("Task 3.2.4: Most frequent ⟨ Noun - Adjective ⟩ pairs for each rating"):
    df = pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    })
with st.expander("Task 3.3: Extraction of Indicative Adjective Phrases"):
    df = pd.DataFrame({
        'first column': [1, 2, 3, 4],
        'second column': [10, 20, 30, 40]
    })


