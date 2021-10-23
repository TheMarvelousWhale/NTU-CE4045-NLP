import streamlit as st

st.set_page_config(page_title='My First App', page_icon=':smiley', layout="wide", initial_sidebar_state='expanded')

from streamlit import session_state
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from spacy_streamlit import visualize_parser
from bs4 import BeautifulSoup
import json
import nltk
import pandas as pd
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
    st.write("")
    st.write("This is the data visualisation of the tasks assigned.")
    st.write("Navigate by selecting the buttons in the sidebar and please wait for the code to run finish before selecting another button. Thank you!")

b1 = st.sidebar.button("1. Tokenization and Stemming", key="1")
b2 = st.sidebar.button("2. POS Tagging", key="2")
b3 = st.sidebar.button("3. Writing Style", key="3")
b4 = st.sidebar.button("4. Most frequent ⟨ Noun - Adjective ⟩ pairs for each rating", key="4")
b5 = st.sidebar.button("5. Extraction of Indicative Adjective Phrases", key="5")


# Load SpaCy models
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
    placeholder.empty()
    st.subheader("POS Tagging:")
    st.write("2 sets of tagging results, 'pos_1' and 'pos_2'")
    df = pd.read_csv("POS_Tag.csv")
    df
    
if b3:
    placeholder.empty()
    st.header("Writing Styles:")
    st.write("Discussion points based on the formality of the way of writing, proper use of English sentence structure such as good grammar, proper pronouns, capitalization, and terms used in the posts.")
    st.write("")
    st.subheader("Stack Overflow - ")
    st.write("Stack Overflow has a relatively less formal way of writing. Informal words like \"Doesn't, I'm, I've...\" are more prevalent that its counterpart. In general, we could observe decent use of English and sentence structure. However, that could be better. In many posts, the start of their sentences and bullet points are not capitalised. Moreover, usage of some words are incorrect e.g. Conjunctions used in the post body \"So I can filter them out..." or "And now we see that using ...\". It is worth noting that there are technical terms and codes used in Stack Overflow that could be Out Of Vocabulary (OOV) words. For example, tree_.getstate() and graphviz' .dot file are used in the posts but the words are OOV. Therefore, tokenization need to be altered to account for unknown tokens (UNKs). Similarly, we should use a software-specific POS tagger to handle the text from Stack Overflow.")
    st.write("")
    st.subheader("Hardware Zone - ")
    st.write("Hardware Zone has two main components - news and forums. The news portion is similar to Channel NewsAsia as most of the news are retrieved from proper news websites, our group will be focusing on the forums. The forums are filled with informal way of writing. The users are mainly Singaporeans, which can explain the common use of Singlish, improper English and structure. On one of the posts - 'any reviews for their boxes? like streams always buffering??', we can observe the lack of capitalisation, use of broken English and exccessive use of punctuation. For another post, it utilised Singlish - 'Change to dark mode lor'. There is also frequent use of emojis or expressions in many posts. Hence, the terms used could be less complex in Hardware Zone as there is very little complex technical terms and unique words used as compared to Stack Overflow.")
    st.write("")
    st.subheader("Channel NewsAsia - ")
    st.write("Channel NewsAsia is a news platform. The journalists are proficient in English and there are many layers of vetting before a post is published. On that account, Channel NewsAsia has a formal way of writing, the most proper use of English sentence structure as compared to the other two counterparts. The posts have proper punctuations and sentence structures. Each sentence is gramatically correct, has capitalisation at the start and proper nouns are capitalised. There might be some unknown terms used such as 'Oxbotica' or 'AppliedEV' but with proper sentence structures, there is enough context to understand. Out of all 3 websites, it would be the easiest to apply tokenization and POS tagging directly to this news posts.")
    st.write("")
    st.write("")
    st.subheader("Sources - ")
    st.markdown("Stack Overflow https://stackoverflow.com/questions/32506951/how-to-explore-a-decision-tree-built-using-scikit-learn , https://stackoverflow.com/questions/3437059/does-python-have-a-string-contains-substring-method")
    st.markdown("Hardware Zone https://forums.hardwarezone.com.sg/threads/new-forum-bugs-reporting-list.6488755/ , https://forums.hardwarezone.com.sg/threads/android-tv-box.5678618/")
    st.markdown("Channel NewsAsia https://www.channelnewsasia.com/business/oxbotica-develop-multi-purpose-self-driving-vehicle-appliedev-2162676 , https://www.channelnewsasia.com/business/new-zealand-banks-post-office-hit-outages-apparent-cyber-attack-2162891")

if b4:
    placeholder.empty()
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
