import random
from DataExploration import process_raw_data
import json, re, random
import spacy

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

def generate_phrase_dict(review_list):
    phrase_dict = {}
    adj_list = ['JJ', 'JJR', 'JJS'] #JJ: Adjective; JJR: comparative adjective; JJS: superlative adjective
    noun_list = ['NN', 'NNS', 'NNP', 'NNPS'] #NN: Singular Noun; #NNS: Plural Noun; #NNP: Singular Proper Noun, #NNPS: Plural Proper Noun
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
                    fin = len(phrase)-index
                    break
            print("")
            if start==-1 or fin==-1:
                print("No adj-noun found")
            else:
                print("Phrase:", phrase[start:fin])
                sub_phrase = str(phrase[start:fin]).lower()
                if sub_phrase in phrase_dict:
                    phrase_dict[sub_phrase] = phrase_dict[sub_phrase]+1
                else:
                    phrase_dict[sub_phrase]=1
            print("")
    return phrase_dict


def generate_phrase_dict_tree(review_list):
    phrase_dict = {}
    for review in review_list:
        doc = nlp_trf(review['text'])
        parsed_list = [0] * len(doc)  # A list used to tracked which tokens have been parsed before.
        for token in doc:
            if parsed_list[token.i] == 1:  # If token has been parsed before, skip this token to prevent double parsing
                continue
            phrase = ""

            # Capture straightforward adjective-nouns
            if token.pos_ == 'ADJ':
                cur_tok = token
                # For chained adjectives
                # e.g. Black-haired lady
                while cur_tok.dep_ == 'amod' and cur_tok.head.pos_ == 'ADJ':
                    parsed_list[cur_tok.i] = 1
                    phrase = phrase + " " + cur_tok.text
                    cur_tok = cur_tok.head
                # Get the noun after finding the necessary adjectives
                if cur_tok.dep_ == 'amod' and cur_tok.head.pos_ == 'NOUN':
                    parsed_list[cur_tok.i] = 1
                    phrase = phrase + " " + cur_tok.text + " " + cur_tok.head.text
                if phrase == '':
                    continue
                phrase = phrase.lower()
                phrase = phrase[1:]  # rid whitespace infront
                if phrase in phrase_dict:
                    phrase_dict[phrase] = phrase_dict[phrase] + 1
                else:
                    phrase_dict[phrase] = 1
            # Capture adjective-nouns separated by AUX, e.g. The Cat is Fat -> fat cat
            if token.pos_ == 'AUX':
                phrase_list = []
                # Search for NOUN/ADJ related to this AUX
                aux_child = [child for child in token.children]
                this_noun = ""
                this_adj = ""
                for child in aux_child:
                    if child.pos_ == 'NOUN':
                        this_noun = child
                    if child.pos_ == 'ADJ':
                        this_adj = child
                # If no adj-noun pair is found, terminate
                if str(this_noun) == "" or str(this_adj) == "":
                    continue
                phrase_list.append(this_adj.text + " " + this_noun.text)
                parsed_list[this_adj.i] = 1
                parsed_list[this_noun.i] = 1
                adj_child = [child for child in this_adj.children]
                child_pos = [child.pos_ for child in adj_child]
                # Traverse adjective's child to see if any other adjective is conjunction with this adjective
                # e.g. Fat and Orange cat -> fat cat, orange cat
                # Recursion not required because there should only be at most 1 ADJ among the children
                while len(adj_child) != 0 and 'ADJ' in child_pos:
                    for child in adj_child:
                        if child.pos_ == 'ADJ':
                            phrase_list.append(child.text + " " + this_noun.text)
                            parsed_list[child.i] = 1
                            adj_child = [grandchild for grandchild in child.children]
                            child_pos = [grandchild.pos_ for grandchild in adj_child]

                for phrase in phrase_list:
                    #                     if phrase == '':
                    #                         print("R2", token.i, token.text,'|\t', token.pos_, token.dep_,'|', token.head.text,'|', token.head.pos_,'|',
                    #                     [child for child in token.children])
                    phrase = phrase.lower()
                    if phrase in phrase_dict:
                        phrase_dict[phrase] = phrase_dict[phrase] + 1
                    else:
                        phrase_dict[phrase] = 1

    return phrase_dict


def get_random_reviews(dict_by_biz, stars, num):
    selected = []
    review_list = []
    count = 0
    while count != num:
        sampled_biz = random.sample(sorted(dict_by_biz), num - count)
        for i in sampled_biz:
            if len(dict_by_biz[i][str(stars)]) == 0:  # If no specific rating found for particular business, skip
                continue
            if i in selected:  # If a review from a particular business have been extracted before
                continue
            count += 1
            selected.append(i)
            random_review = random.sample(dict_by_biz[i][str(stars)], 1)[0]
            review_list.append(random_review)
    return review_list

# Populate individual list by stars
stars_1 = get_random_reviews(business_dict,1, 50)
stars_2 = get_random_reviews(business_dict,2, 20)
stars_3 = get_random_reviews(business_dict,3, 20)
stars_4 = get_random_reviews(business_dict,4, 20)
stars_5 = get_random_reviews(business_dict,5, 20)

stars_1_pt = generate_phrase_dict_tree(stars_1)
stars_2_pt = generate_phrase_dict_tree(stars_2)
stars_3_pt = generate_phrase_dict_tree(stars_3)
stars_4_pt = generate_phrase_dict_tree(stars_4)
stars_5_pt = generate_phrase_dict_tree(stars_5)

print("1 Star")
sorted_dict = {k: v for k, v in sorted(stars_1_pt.items(), key=lambda item: item[1], reverse=True)}.items()
print(list(sorted_dict)[:10])
print("2 Star")
sorted_dict = {k: v for k, v in sorted(stars_2_pt.items(), key=lambda item: item[1], reverse=True)}.items()
print(list(sorted_dict)[:10])
print("3 Star")
sorted_dict = {k: v for k, v in sorted(stars_3_pt.items(), key=lambda item: item[1], reverse=True)}.items()
print(list(sorted_dict)[:10])
print("4 Star")
sorted_dict = {k: v for k, v in sorted(stars_4_pt.items(), key=lambda item: item[1], reverse=True)}.items()
print(list(sorted_dict)[:10])
print("5 Star")
sorted_dict = {k: v for k, v in sorted(stars_5_pt.items(), key=lambda item: item[1], reverse=True)}.items()
print(list(sorted_dict)[:10])

# Show dependency graph
doc = nlp_trf("this is a very fat and orange cat")
for token in doc:
    print(token.text,'|',token.dep_,'|', token.head.text,'|', token.head.pos_,'|',
            [child for child in token.children])
spacy.displacy.render(doc, style='dep')

