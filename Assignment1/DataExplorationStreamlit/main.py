import streamlit as st

st.set_page_config(page_title='NLP Assignment 1', page_icon=':smiley', layout="wide", initial_sidebar_state='expanded')

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
import NounAdjPair

st.sidebar.header("CE4045 NLP Assignment 1:")
st.sidebar.write("Please select from the sidebar to navigate")

placeholder = st.empty()

with placeholder.container():
    st.header("First Assignment for NLP:")
    st.subheader("This project is done by: Kheng Quan, Viet, Shan Jie, Chia Yu, and Hao Wei.")
    st.write("")
    st.write("This is the data visualisation and discussion of the tasks assigned, where the segments for task 1 - 5 are separated into data analysis then discussion.")
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
    st.header("Tokenization and Stemming:")
    st.subheader("Data - 1st chosen business:")
    DataExploration.analyze_business(chosen_business_1)

    st.subheader("Data - 2nd chosen business:")
    DataExploration.analyze_business(chosen_business_2)
    st.write("")
    st.subheader("Discussion")
    st.write("This solution builds a word frequency dictionary for 2 randomly selected businesses")
    st.write("We can observe that the top most common words are unchanged after stemming. This is a good indication that the top words are words of different forms e.g. plural vs singular. Stemming will reduce the inflectional forms of the word into 1 single token, which is likely better for word counting. For example, services and servicing will be merged with service, increasing the word count of servic(e).")
    st.write("It is worth noting that stemming would merge different senses of homonyms as well. However, from the graph, the loss in sense of homonyms is not as significant as there is no drastic change to the frequency count.")


if b2:
    st.header("Part-of-Speech Tagging:")
    placeholder.empty()
    st.subheader("Data")
    st.write("2 sets of tagging results, 'pos_1' and 'pos_2'")
    df = pd.read_csv("../POS_Tag.csv")
    df = df.iloc[:, :-1]
    df
    st.subheader("Discussion")
    st.markdown("This analysis used **pos_1** as **fine grain POS tag** while **pos_2** as **coarse grain POS tag**.")
    st.write("It can be observed that there are vast similarities between pos_1 and pos_2 e.g. The token - \"really\" is tagged as RB and ADV, which both stand for adverbs. On contrary, differences between pos_1 (fine grain) and pos_2 (coarse grain) like \"that\" being WDT (Wh-determiner) and \"the\" being DT (determiner) for pos_1 but both being DET for pos_2 could both be useful depending on the context of dataset given.")
    st.write("In our case where the data analysis is done on restaurant reviews, pos_2 might serve to be more useful as the analysis require less precision during data processing as compared to datasets with higher formality e.g. being used in a collection of reports. This is due to tagging of distinct determiners, such as \"that\" or \"the\", being less significant for the purpose of our data analysis. Using the coarse grain POS tag would be suffice in most data analysis contexts unless the task requires a more specific tags, then fine grain POS tag could be deployed.")

    
if b3:
    placeholder.empty()
    st.header("Writing Styles:")
    st.write("Discussion points are based on the formality of the way of writing, proper use of English sentence structure such as good grammar, proper pronouns, capitalization, and terms used in the posts.")
    st.write("")
    st.subheader("Stack Overflow")
    st.write("Stack Overflow has a relatively less formal way of writing. Informal words like \"Doesn't, I'm, I've...\" are more prevalent that formal words. In general, we could observe decent use of proper English and sentence structure. However, that could be improved. In many posts, the start of the sentences and bullet points are not capitalised. Moreover, usage of some words are incorrect e.g. Conjunctions used in the post body \"So I can filter them out...\" or \"And now we see that using ...\". It is worth noting that there are technical terms and codes used in Stack Overflow that could be Out Of Vocabulary (OOV) words. For example, tree_.getstate() and graphviz' .dot file are used in the posts but the words are OOV. Therefore, tokenization need to be altered to account for unknown tokens (UNKs). Similarly, we should use a software-specific POS tagger to handle the text from Stack Overflow.")
    st.write("")
    st.subheader("Hardware Zone")
    st.write("Hardware Zone has two main components - news and forums. The news portion is similar to Channel NewsAsia as most of the news are retrieved from proper news websites, thus our group will be focusing on the forums. The forums are filled with informal way of writing. The users are mainly Singaporeans, which can explain the common use of Singlish, improper English and structure. On one of the posts - \"any reviews for their boxes? like streams always buffering??\", we can observe the lack of capitalisation, use of broken English and exccessive use of punctuations. For another post, it utilises Singlish - \"Change to dark mode lor\". There is also frequent use of emojis or expressions in many posts. Hence, the terms used will be different in Hardware Zone than Stack Overflow as there is very need for overly complex technical terms and unique words specific for coding for Stack Overflow.")
    st.write("")
    st.subheader("Channel NewsAsia")
    st.write("Channel NewsAsia is a news platform. The journalists are expected to be proficient in English and there are many layers of vetting before a post is published. On that account, Channel NewsAsia has a formal way of writing, the most proper use of English sentence structure as compared to the other two counterparts. The posts have proper punctuations and sentence structures. Each sentence is grammatically correct, has capitalisation at the start and proper nouns are capitalised. There might be some unknown technical terms used such as 'Oxbotica' or 'AppliedEV' but with proper sentence structures, there will be enough context to understand. Out of all 3 websites, it would be the easiest to apply tokenization and POS tagging directly to this news posts.")
    st.write("")
    st.write("")
    st.subheader("Sources")
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
    st.header("Top-10 most frequent noun-adjective pairs:")
    st.subheader("For 1 - 5 stars:")
    st.write("Data visualization, only 10 pairs are shown per rating.")
    NounAdjPair.print_sorted_dict(stars_1_pt, stars_2_pt, stars_3_pt, stars_4_pt, stars_5_pt)

    st.subheader("Discussion")
    st.write("This extraction identified the common characteristics of a group of reviews within their own rating, which would be done across 1 - 5 stars. 50 random businesses were extracted for 1 star reviews, while 20 random businesses were extracted for 2 - 5 stars reviews.")
    st.write("An issue with noun-adjective pairs is that the corresponding adjectives might not adjacent to one another but may appear before or after the noun, e.g. The service is good vs good service. Hence, a set of rules were formulated to extract the Noun-Adjective pairs using the CG POS tag of each token. The complete pairs were added into a dictionary, which will be used to track the frequency of each pair.")
    st.write("The first extraction rule is targeted at cases where the adjective appears before the noun while the second extraction rule is targeted at cases where the adjective appears after the noun. Thus, the lists generated could be useful to further filter the common differences in words used between the different star ratings. For example, good food is commonly used amongst the 4 - 5 stars reviews while bad taste or low quality is commonly used in 1 - 2 stars reviews.")
    st.write("It is worthy to take note that there might be positives words captured by the extraction in poor reviews e.g. great service in 1 star. However, given the context, it could be used to associate with bad service i.e. not a great service.")

if b5:
    placeholder.empty()
    st.header("Extraction of Indicative Adjective Phrases:")
    st.subheader("Usage of adjective pair extractor and tf-idf to obtain the IAPs")

    indicative_phrase = {}
    with open('../indicative_phrase-ltn.json', 'r') as fp:
        indicative_phrase = json.load(fp)
    indicative_phrase

    st.subheader("Discussion")
    st.write("The main advantage of using tf-idf based method is that the indicative phrase for huge dataset of reviews could be extracted unsupervised.")
    st.write("However, in our task, this approach held the assumption that phrase uniqueness of a piece of restaurant review would be a good indicator for the other reviews. This means that all the extracted adjective of that 1 review is treated equally and applied onto other reviews. This way of analysis might not be accurate when it comes to restaurant reviews as there are different aspects of a restaurant like service, food quality or cleanliness that people could critique on. The selected review could be biased to one aspect and not focus or mention on the other aforementioned aspects. This would then result to numerous false positives and little false negatives, leading to low precision yet high recall.")
