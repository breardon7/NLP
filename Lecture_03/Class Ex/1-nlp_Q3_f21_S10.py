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

for token in doc[0:100]:
    print(token.text, token.pos_)



# --------------Find all the words related to dates and time--------------------------------------


for ent in doc.ents:
    if ent.label_ == DATE:
        print(ent)



# --------------Find all URLS--------------------------------------

for token in doc:
    if token.like_url:
        print(token)





# ----------------Document Similarities between ech sentences----------------------

'''text2 = open('data1.txt').read()
print(text.similarity(text2))'''




