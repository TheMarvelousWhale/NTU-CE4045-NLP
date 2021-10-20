import streamlit as st
st.set_page_config(page_title='My First App', page_icon=':smiley', layout="wide", initial_sidebar_state='expanded')
b1 = st.sidebar.button("1. Tokenization and Stemming", key="1")
b2 = st.sidebar.button("2. POS Tagging", key="2")
b3 = st.sidebar.button("3. Writing Style", key="3")
b4 = st.sidebar.button("4. Most frequent ⟨ Noun - Adjective ⟩ pairs for each rating", key="4")
b5 = st.sidebar.button("5. Extraction of Indicative Adjective Phrases", key="5")

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
import POSTagging
import NounAdjPair
import IndicativeAdjectivePhrases
"""
# My first app
CE4045 NLP Assignment 1:
"""
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


if b1:
    chosen_business_1, chosen_business_2 = DataExploration.choosing_bus_id()
    DataExploration.analyze_business(chosen_business_1)
    DataExploration.analyze_business(chosen_business_2)
    # Collecting all unique the business id
    # business_id_list = list(
    #     {x['business_id'] for x in big_json})  # make it a set via set comprehension {}, then call tolist
    #Selecting a random business and their review
    # chosen_id_1 = random.choice(business_id_list)
    # chosen_business_1 = [x for x in big_json if x['business_id'] == chosen_id_1]
    # chosen_id_2 = chosen_id_1
    # while chosen_id_2 == chosen_id_1:
    #     chosen_id_2 = random.choice(business_id_list)
    #     chosen_business_2 = [x for x in big_json if x['business_id'] == chosen_id_2]
    # DataExploration.analyze_business(chosen_business_1)
    # DataExploration.analyze_business(chosen_business_2)
    # exec(open("DataExploration.py").read())

if b2:
    POSTagging.print_POS_solution()
    
if b3:
    st.write("Discussion points based on the formality of the way of writing, proper use of English sentence structure such as good grammar, proper pronouns, capitalization, and terms used in the posts.")
    st.subheader("Stack Overflow")

if b4:
    print("bye")
    business_dict = {}
    main_compile_list = []
    #Restructure big_json to group by business & the stars
    business_dict = NounAdjPair.return_bus_dict()
    #Populate individual list by stars
    stars_1 = NounAdjPair.get_random_reviews(business_dict, 1, 50)
    stars_2 = NounAdjPair.get_random_reviews(business_dict, 2, 20)
    stars_3 = NounAdjPair.get_random_reviews(business_dict, 3, 20)
    stars_4 = NounAdjPair.get_random_reviews(business_dict, 4, 20)
    stars_5 = NounAdjPair.get_random_reviews(business_dict, 5, 20)

    stars_1_pt = NounAdjPair.generate_phrase_dict_tree(stars_1)
    stars_2_pt = NounAdjPair.generate_phrase_dict_tree(stars_2)
    stars_3_pt = NounAdjPair.generate_phrase_dict_tree(stars_3)
    stars_4_pt = NounAdjPair.generate_phrase_dict_tree(stars_4)
    stars_5_pt = NounAdjPair.generate_phrase_dict_tree(stars_5)

    main_compile_list = NounAdjPair.print_sorted_dict(stars_1_pt, stars_2_pt, stars_3_pt, stars_4_pt, stars_5_pt)
    NounAdjPair.print_compiled_list(main_compile_list)
    NounAdjPair.show_dep_graph()

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
    #try only showing 10
    indicative_phrase

    #Show dependency graph
    IndicativeAdjectivePhrases.show_iap_dep_graph()

