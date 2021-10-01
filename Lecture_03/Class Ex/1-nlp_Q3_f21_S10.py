# ------------------Import Library----------------------------
import spacy


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

fig, ax = plt.subplots()
im = ax.imshow(sim_array)

ax.set_title("Sentence Similarity")
fig.tight_layout()
plt.show()

# The heatmap is a bit difficult to read given the scale, but the yellow diagonal line represents
# the projection of a sentence onto itself, scoring a 1 which is the highest similarity score. This
# indicates that lighter cells equal a higher score in the plot, and darker equals a lower score.
# Based on that info, it seems most sentences have a relatively high similarity with one another,
# except the following sentences: 18, 51, 54.