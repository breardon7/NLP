# ------------------Import Library----------------------------
import spacy
import pandas as pd


#-------------------Load dataset and Tokenize------------------------------
text = open('data.txt').read()
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
for token in doc:
    print(token)





# -------------- Part of speech tags--------------------------------------
text = open('clean_data.txt').read()
doc = nlp(text)
for token in doc[0:100]:
    print(token.text, token.pos_)



# --------------Find all the words related to dates and time--------------------------------------


for ent in doc.ents:
    if ent.label_ == 'DATE' or ent.label_ == 'TIME':
        print(ent)



# --------------Find all URLS--------------------------------------

for token in doc:
    if token.like_url:
        print(token)




# ----------------Document Similarities between ech sentences----------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
text2 = open('data1.txt', 'r').read()

doc = nlp(text2)
sentences = [s for s in doc.sents]

list2 = []
for sent in doc.sents:
    list1 = []
    for sent2 in doc.sents:
        list1.append(sent.similarity(sent2))
    list2.append(list1)
sim_array = np.array(list2)
print(sim_array)
print(list2)

sns.set(font_scale = 4)
plt.figure(figsize=(100,75))
sns.heatmap(sim_array)
plt.show()
