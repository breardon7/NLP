# E.1
# i.
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
print(df.columns)
author = df['author']

# ii.
import spacy
nlp = spacy.load("en_core_web_sm")
string = df['title'][0]
doc = nlp(string)
tokens = [token.text for token in doc]
index = [token.i for token in doc]
punctuation = [token.is_punct for token in doc]
lemma = [token.lemma_ for token in doc]
POS = [token.pos_ for token in doc]

frame = {'tokens': tokens, 'index': index, 'punctuation': punctuation, 'lemma': lemma, 'POS': POS}
df_tokens = pd.DataFrame(frame)
print(df_tokens.head())

# iii.
for token in doc.ents:
    print(token)
# iv.
string = df['title'][1]
doc = nlp(string)
for token in doc:
    if token.pos_ == 'NOUN':
        print(token)
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,
            chunk.root.head.text)
# v.
for token in doc:
    print(token.text, token.dep_, token.head.text)
# vi.

# E.2
# i.
# ii.
# iii.

# E.2
# i.
# ii.
# iii.
# iv.