import random
from DataExploration import process_raw_data
import json, re, random
import spacy

big_data_file = './reviewSelected100.json'

big_data = process_raw_data(big_data_file)

big_json = [json.loads(x) for x in big_data]

nlp_trf = spacy.load("en_core_web_trf")


def generate_phrase_dict(review_list):
    phrase_dict = {}
    adj_list = ['JJ', 'JJR', 'JJS']  # JJ: Adjective; JJR: comparative adjective; JJS: superlative adjective
    noun_list = ['NN', 'NNS', 'NNP',
                 'NNPS']  # NN: Singular Noun; #NNS: Plural Noun; #NNP: Singular Proper Noun, #NNPS: Plural Proper Noun
    for review in review_list:
        doc = nlp_trf(review['text'])
        for phrase in doc.noun_chunks:
            start = -1
            fin = -1
            print(phrase, end=" | ")
            for index, i in enumerate(phrase):
                print(i.tag_, end=" | ")
                if i.tag_ in adj_list and start == -1:
                    start = index
            for index, i in enumerate(reversed(phrase)):
                if i.tag_ in noun_list and fin == -1:
                    fin = len(phrase) - index
                    break
            print("")
            if start == -1 or fin == -1:
                print("No adj-noun found")
            else:
                print("Phrase:", phrase[start:fin])
                sub_phrase = str(phrase[start:fin]).lower()
                if sub_phrase in phrase_dict:
                    phrase_dict[sub_phrase] = phrase_dict[sub_phrase] + 1
                else:
                    phrase_dict[sub_phrase] = 1
            print("")
    return phrase_dict


# Populate individual list by stars
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

random_reviews = random.sample(stars_1, 5)
phrase_dict_1 = generate_phrase_dict(random_reviews)

phrase_dict_1
