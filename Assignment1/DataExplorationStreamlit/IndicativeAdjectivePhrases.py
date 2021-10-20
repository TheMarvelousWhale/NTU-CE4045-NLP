# To use pre-generated, import json instead. DO NOT NEED TO RE-RUN
# Generate all business phrases
import random
import pandas as pd
import numpy as np
from DataExploration import process_raw_data
import json, re, random
import spacy
from NounAdjPair import generate_phrase_dict_tree

big_data_file = './reviewSelected100.json'

big_data = process_raw_data(big_data_file)

big_json = [json.loads(x) for x in big_data]

nlp_trf = spacy.load("en_core_web_trf")

# Restructure big_json to group by business & the stars
business_dict = {}
for i in big_json:
    if business_dict.get(i['business_id'])==None:
        business_dict[i['business_id']] = {}
        business_dict[i['business_id']]['1'] = []
        business_dict[i['business_id']]['2'] = []
        business_dict[i['business_id']]['3'] = []
        business_dict[i['business_id']]['4'] = []
        business_dict[i['business_id']]['5'] = []
    if i['stars']==1:
        business_dict[i['business_id']]['1'].append(i)
    elif i['stars'] == 2:
        business_dict[i['business_id']]['2'].append(i)
    elif i['stars'] ==3:
        business_dict[i['business_id']]['3'].append(i)
    elif i['stars'] == 4:
        business_dict[i['business_id']]['4'].append(i)
    else:
        business_dict[i['business_id']]['5'].append(i)

biz_phrases = {}
for index, i in enumerate(business_dict):
    print(index, i)
    biz_review = []
    for j in range(1, 6):
        biz_review.extend(business_dict[i][str(j)])
    biz_phrases[i] = generate_phrase_dict_tree(biz_review)

with open('business_phrase.json', 'w') as fp:
    json.dump(biz_phrases, fp)

# Load Pre-generated json
with open('business_phrase.json', 'r') as fp:
    biz_phrases=json.load(fp)

tf = pd.DataFrame()
for biz in biz_phrases:
    for phrase in biz_phrases[biz]:
        tf.at[phrase, biz] = biz_phrases[biz][phrase]
tf = tf.fillna(0) # Replace all NaN with 0

num_of_biz = len(tf.columns)
idf = []
for i in (tf == 0).astype(int).sum(axis=1): # Generate the list of all IDF
    term_idf = np.log10(num_of_biz/(num_of_biz-i))
    idf.append(term_idf)

tf = tf.mul(idf, axis=0) # Multiply each Term Frequency with the Inversed Document Frequency
tf.to_csv("tfidf.csv") # Export TF-IDF table

indicative_phrase = {}
for biz in biz_phrases:
    indicative_phrase[biz]={}
    indicate_phrase_pos = tf[biz].argmax()
    indicative_phrase[biz]['phrase']=tf.index[indicate_phrase_pos]
    indicative_phrase[biz]['tfidf'] = tf[biz][indicate_phrase_pos]
with open('indicative_phrase.json', 'w') as fp:
    json.dump(indicative_phrase, fp)