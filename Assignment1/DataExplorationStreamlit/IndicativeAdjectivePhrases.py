# To use pre-generated, import json instead. DO NOT NEED TO RE-RUN
# Generate all business phrases
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataExploration import process_raw_data
import json, re, random
import spacy
from NounAdjPair import generate_phrase_dict_tree
import streamlit as st

nlp_trf = spacy.load("en_core_web_trf")

# Restructure big_json to group by business & the stars
def get_adjective_phrase(review_list):
    phrase_dict = {}
    for review in review_list:
        print(review['text'])
        doc = nlp_trf(review['text'])
        for token in doc:
            cur_token = token
            bound_start = 0
            bound_end = 0
            if cur_token.pos_ == 'ADJ':
                bound_start = cur_token.i
                bound_end = cur_token.i + 1
                children_queue = [child for child in cur_token.children]
                while len(children_queue) != 0:
                    child = children_queue[0]
                    if child.pos_ == 'CCONJ' or child.pos_ == 'SCONJ':
                        break
                    if child.dep_ == 'advmod':
                        if child.i < bound_start:
                            bound_start = child.i
                    if child.i >= bound_end:
                        bound_end = child.i + 1
                    grand_children = [grandchild for grandchild in child.children]
                    children_queue.extend(grand_children)
                    children_queue.pop(0)
            phrase = doc[bound_start:bound_end].text
            phrase = phrase.lower()
            phrase = re.sub(r'\W+', ' ', phrase)
            if phrase == "":
                continue
            print(phrase)
            if phrase in phrase_dict:
                phrase_dict[phrase] = phrase_dict[phrase] + 1
            else:
                phrase_dict[phrase] = 1

    return phrase_dict

