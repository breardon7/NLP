# =================================================================
# Class_Ex1:
# Import spacy abd from the language class import english.
# Create a doc object
# Process a text : This is a simple example to initiate spacy
# Print out the document text from the doc object.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')

import spacy
nlp = spacy.load("en_core_web_sm")
text = 'this is a test'
doc = nlp(text)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)

print(20*'-' + 'End Q1' + 20*'-')
# =================================================================
# Class_Ex2:
# Solve Ex1 but this time use German Language.
# Grab a sentence from german text from any website.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')

nlp = spacy.load("de_core_news_sm")
text = 'Jeder hat das Recht auf Bildung. Die Bildung ist unentgeltlich, zum mindesten der Grundschulunterricht und die grundlegende Bildung.'
doc = nlp(text)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)


print(20*'-' + 'End Q2' + 20*'-')
# =================================================================
# Class_Ex3:
# Tokenize a sentence using sapaCy.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')

import spacy
nlp = spacy.load("en_core_web_sm")
text = 'this is a test'
doc = nlp(text)
for token in doc:
    print(token.text)

print(20*'-' + 'End Q3' + 20*'-')
# =================================================================
# Class_Ex4:
# Use the following sentence as a sample text. and Answer the following questions.
# "In 2020, more than 15% of people in World got sick from a pandemic ( www.google.com ). Now it is less than 1% are. Reference ( www.yahoo.com )"
# 1- Check if there is a token resemble a number.
# 2- Find a percentage in the text.
# 3- How many url is in the text.

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q4' + 20*'-')

text = "In 2020, more than 15% of people in World got sick from a pandemic ( www.google.com ). Now it is less than 1% are. Reference ( www.yahoo.com )"
doc = nlp(text)
nums = [t.like_num for t in doc]
nums_count = nums.count(True)
print(nums_count/len(doc)*100)
for token in doc:
    if token.like_url:
        print(token)

print(20*'-' + 'End Q4' + 20*'-')
# =================================================================
# Class_Ex5:
# Load small web english model into spaCy.
# USe the following text as a sample text. Answer the following questions
# "It is shown that: Google was not the first search engine in U.S. tec company. The value of google is 100 billion dollar"
# 1- Get the token text, part-of-speech tag and dependency label.
# 2- Print them in a tabular format.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q5' + 20*'-')

nlp = spacy.load("en_core_web_sm")
text = "It is shown that: Google was not the first search engine in U.S. tec company. The value of google is 100 billion dollar"
doc = nlp(text)
for token in doc:
    print(token.text, token.pos_, token.dep_)


print(20*'-' + 'End Q5' + 20*'-')

# =================================================================
# Class_Ex6:
# Use Ex 5 sample text and find all the entities in the text.

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q6' + 20*'-')

for token in doc.ents:
    print(token.text, token.label_)

print(20*'-' + 'End Q6' + 20*'-')
# =================================================================
# Class_Ex7:
# Use SpaCy and find adjectives plus one or 2 nouns.
# Use th efollowinf Sample text.
# Features of the iphone applications include a beautiful design, smart search, automatic labels and optional voice responses.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q7' + 20*'-')

text = 'Features of the iphone applications include a beautiful design, smart search, automatic labels and optional voice responses.'
doc = nlp(text)
for token in doc:
    if token.pos_ == 'ADJ':
        print(token.text, token.pos_)
for token in doc:
    if token.pos_ == 'NOUN':
        print(token.text, token.pos_)
        break
print(20*'-' + 'End Q7' + 20*'-')
# =================================================================
# Class_Ex8:
# Use spacy lookup table and find the hash id for a cat
# Text : I have a cat.
# Next use the id and and find the string.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q8' + 20*'-')

nlp = spacy.load('en_core_web_sm')
nlp.vocab.strings.add('I have a cat.')
cat_hash = nlp.vocab.strings['I have a cat.']
cat_string = nlp.vocab.strings[cat_hash]
print(cat_string)


print(20*'-' + 'End Q8' + 20*'-')
# =================================================================
# Class_Ex9:
# Create a Doc object for the following sentence
# Spacy is a nice toolkit.
# Use the methods like text, token,... on the Doc and check the functionality.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q9' + 20*'-')

doc = nlp('Spacy is a nice toolkit.')
print(doc.text, doc.doc, doc.ents, doc.vocab)

print(20*'-' + 'End Q9' + 20*'-')
# =================================================================
# Class_Ex10:
# Use spacy and process the following text.
# Newyork looks like a nice city.
# Find which token is proper noun and which one is a verb.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q10' + 20*'-')

text = 'Newyork looks like a nice city.'
doc = nlp(text)
for token in doc:
    if token.pos_ == 'PROPN' or token.pos_ == 'VERB':
        print(token.text, token.pos_)

print(20*'-' + 'End Q10' + 20*'-')
# =================================================================
# Class_Ex11:
# Read the list of countries in a json format.
# Use the following text as  sample text.
# Czech Republic may help Slovakia protect its airspace
# Use statistical method and rule based method to find the countries.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q11' + 20*'-')

print('---rule-based method---')
import json
file = open('countries.json',)
countries = json.load(file)
text = 'Czech Republic may help Slovakia protect its airspace'
list = text.split(sep=' ')
for word in list:
    if word in countries:
        print(word)

print('---statistical method---')
import json
file = open('countries.json',)
countries = json.load(file)
text = 'Czech Republic may help Slovakia protect its airspace'
doc = nlp(text)
for ent in doc.ents:
    print(ent.text)


print(20*'-' + 'End Q11' + 20*'-')
# =================================================================
# Class_Ex12:
# Use spacy attributions and answer the following questions.
# Define the getter function that takes a token and returns its reversed text.
# Add the Token property extension "reversed" with the getter function
# Process the text and print the results.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q12' + 20*'-')

from spacy.tokens import Token
import spacy
nlp = spacy.load("en_core_web_sm")

def get_reversed_token(token):
    token = str(token)
    reversed = token[::-1]
    return reversed
Token.set_extension('reversed', getter=get_reversed_token)

doc = nlp('Reverse this text')
for token in doc:
    print(token.text, '--', token._.reversed)

print(20*'-' + 'End Q12' + 20*'-')
# =================================================================
# Class_Ex13:
# Read the tweets json file.
# Process the texts and print the entities
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q13' + 20*'-')
import json
file = open('tweets.json',)
tweets = json.load(file)
for string in tweets:
    doc = nlp(string)
    for token in doc.ents:
        print(token)

print(20*'-' + 'End Q13' + 20*'-')
# =================================================================
# Class_Ex14:
# Use just spacy tokenization. for the following text
# "Burger King is an American fast food restaurant chain"
# make sure other pipes are disabled and not used.
# Disable parser and tagger and process the text. Print the tokens
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q14' + 20*'-')

nlp = spacy.load("en_core_web_sm")
with nlp.disable_pipes("tagger", "parser"):
    tokenizer = nlp.tokenizer
text = "Burger King is an American fast food restaurant chain"
tokens = tokenizer(text)
for token in tokens:
    print(token)

print(20*'-' + 'End Q14' + 20*'-')

# =================================================================


