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
import streamlit as st



#preparing the tools

st.set_option('deprecation.showPyplotGlobalUse', False)

ps = PorterStemmer()
lz = WordNetLemmatizer()
en_stopwords = stopwords.words('english')

# Run if SpaCy model hasn't been downloaded before

# Load SpaCy models
nlp_sm = spacy.load("en_core_web_sm")
nlp_trf = spacy.load("en_core_web_trf")

# Need to run if stopwords not downloaded before.
### UNCOMMENT THIS WHEN SUBMITTING ###
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

sample_file = './reviewSamples20.json'
# Clean data
def clean_text(sum_string):
    s = re.sub('[\r\n\s]+',' ',sum_string) #clean whitespce and newline
    return s

#data importing and formating
def process_raw_data(data_in_weird_format_file):
    with open(data_in_weird_format_file,'r') as f:
        raw_data = f.read()
    data = raw_data.split('}\n')
    return [x+'}' for x in data if x != '']
data = process_raw_data(sample_file)
print(f'Processed {len(data)} lines of data')

_sample_data = json.loads(random.choice(data))
print('\n\nSample data')
print(json.dumps(_sample_data, indent=4, sort_keys=True))

#Check if they have the same keys
json_list = [json.loads(x) for x in data]
for i,_json in enumerate(json_list):
    print(f"{i}\t{_json.keys()}")

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
#Because we have to repeat the whole thing for 2 businesses, we need to wrap the functions into a single wrapper for reuse

for y in ['useful','stars','funny','cool']:
    print({y:Counter(x[y] for x in big_json)})
    print()


# Wrapper for analysis  -- not too sure if the inside functions should throw outside anot it's quite task specific
def analyze_business(chosen_business):
    # collating all reviews into one string
    all_dem_reviews = ''.join([x['text'] for x in chosen_business])

    all_dem_clean_reviews = clean_text(all_dem_reviews)

    # for word distribution, we need to clean all punctuations and symbols as well as standardize the case
    def build_word_frequency(text, blacklist=[], stemmer=None):
        assert type(text) == str, "Expecting a str input"
        s = re.sub('[^a-z]+', ' ', text.lower())
        s = re.sub('[\n\r\s]+', ' ', s)
        vocab = Counter()
        for word in s.split(' '):
            if word in blacklist:
                continue
            if stemmer != None:
                word = stemmer.stem(word)
            vocab.update([word])  # preventing the counter object to update using each letter by using []
        return vocab

    review_vocab_bef_stem = build_word_frequency(all_dem_clean_reviews, blacklist=en_stopwords)
    review_vocab_after_stem = build_word_frequency(all_dem_clean_reviews, blacklist=en_stopwords, stemmer=ps)

    st.write("Before stemming, most 10 common words:")
    for x in review_vocab_bef_stem.most_common(10):
        st.write('\t', x[0].ljust(15, ' '), x[1])  # justify the column abit

    st.write('\n\n')

    st.write("After stemming, most 10 common words:")
    for x in review_vocab_after_stem.most_common(10):
        st.write('\t', x[0].ljust(15, ' '), x[1])

    # Plotting the word distribution
    def plot_word_freq_dist(the_vocab):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        line = ax.plot(the_vocab.values())

        ax.set_yscale('log')
        st.pyplot(fig)

    plot_word_freq_dist(review_vocab_after_stem)

