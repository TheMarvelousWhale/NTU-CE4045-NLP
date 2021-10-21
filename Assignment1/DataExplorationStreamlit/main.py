import streamlit as st

st.set_page_config(page_title='My First App', page_icon=':smiley', layout="wide", initial_sidebar_state='expanded')

from streamlit import session_state
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import json,re,random
import matplotlib.pyplot as plt
import spacy
from spacy_streamlit import visualize_parser
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

st.sidebar.header("CE4045 NLP Assignment 1:")
st.sidebar.write("Please select from the sidebar to navigate")

placeholder = st.empty()

with placeholder.container():
    st.header("First Assignment for NLP:")
    st.subheader("This project is done by: Kheng Quan, Viet, Shan Jie, Chia Yu, Hao Wei.")
    st.write()
    st.write("This is the data visualisation of the tasks assigned")
    st.write("Navigate by selecting the buttons in the sidebar and please wait for the code to run finish before selecting another button. Thank you!")

b1 = st.sidebar.button("1. Tokenization and Stemming", key="1")
b2 = st.sidebar.button("2. POS Tagging", key="2")
b3 = st.sidebar.button("3. Writing Style", key="3")
b4 = st.sidebar.button("4. Most frequent ⟨ Noun - Adjective ⟩ pairs for each rating", key="4")
b5 = st.sidebar.button("5. Extraction of Indicative Adjective Phrases", key="5")


# Load SpaCy models
# nlp_sm = spacy.load("en_core_web_sm")
nlp_trf = spacy.load("en_core_web_trf")

# Need to run if stopwords not downloaded before.
### UNCOMMENT THIS WHEN SUBMITTING ###
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# Need to run if stopwords not downloaded before.


if b1:
    placeholder.empty()
    col1, col2 = st.columns(2)
    chosen_business_1, chosen_business_2 = DataExploration.choosing_bus_id()
    with col1:
        col1.subheader("Before Stemming:")
        DataExploration.analyze_business(chosen_business_1)

    with col2:
        col2.subheader("After Stemming:")
        DataExploration.analyze_business(chosen_business_2)

if b2:
    st.subheader("POS Tagging:")
    st.write("2 sets of tagging results, 'pos_1' and 'pos_2'")
    df = pd.read_csv("POS_Tag.csv")
    df
    
if b3:
    st.write("Discussion points based on the formality of the way of writing, proper use of English sentence structure such as good grammar, proper pronouns, capitalization, and terms used in the posts.")
    st.subheader("Stack Overflow")

if b4:
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
    st.subheader("Top-10 most frequent noun-adjective pairs:")
    st.write("For 1 - 5 stars:")
    NounAdjPair.print_sorted_dict(stars_1_pt, stars_2_pt, stars_3_pt, stars_4_pt, stars_5_pt)

    # printer_compile_list = NounAdjPair.print_compiled_list(main_compile_list)
    # printer_compile_list
    #doc = nlp_trf("this is a very fat and orange cat")
    # doc = nlp_trf("this is a very fat and orange cat")
    # for token in doc:
    #     st.write(token.text, '|', token.dep_, '|', token.head.text, '|', token.head.pos_, '|',
    #              [child for child in token.children])
    # spacy.displacy.render(doc, style='dep')
    # visualize_parser(doc)

if b5:
    st.subheader("Extraction of Indicative Adjective Phrases:")
    st.write("Usage of adjective pair extractor and tf-idf to obtain the IAPs -")
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

    # doci = nlp_trf("The cat is fat and fluffy")
    # for token in doci:
    #     st.write(token.text, '|', token.dep_, '|', token.head.text, '|', token.head.pos_, '|',
    #              [child for child in token.children])
    # visualize_parser(doci)
    # Show dependency graph
    # IndicativeAdjectivePhrases.show_iap_dep_graph()
    # tf = pd.read_csv("tf.csv")
    # s = tf.sum(axis=1)
    # t = s.sort_values(ascending=False)
    # t = t.reset_index()
    # temph = t[0][:500]
    #
    # plt.bar(t.index[:500],temph)
    # st.pyplot()

