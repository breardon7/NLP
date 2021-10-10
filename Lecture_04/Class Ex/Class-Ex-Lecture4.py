# =================================================================
# Class_Ex1:
# Write a function that checks a string contains only a certain set of characters
# (all chars lower and upper case with all digits).
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')
text = 'This is a test string 123^&#@*98'
string = 'Teststring123'
def check_string(text):
    import re
    pattern = re.compile(r'\w')
    matches = pattern.finditer(text)
    count = 0
    for i in matches:
        count += 1
    if count == len(text):
        print('String is valid')
    else:
        print('String is not valid')

print(check_string(text))
print(check_string(string))



print(20*'-' + 'End Q1' + 20*'-')

# =================================================================
# Class_Ex2:
# Write a function that matches a string in which a followed by zero or more b's.
# Sample String 'ac', 'abc', abbc'
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')
s1 = 'ac'
s2 = 'abc'
s3 = 'abbc'
def zero_or_more_b(text):
    import re
    pattern = re.compile(r'^ab*')
    matches = pattern.finditer(text)
    for match in matches:
        print(match)

print(zero_or_more_b(s1))
print(zero_or_more_b(s2))
print(zero_or_more_b(s3))


print(20*'-' + 'End Q2' + 20*'-')
# =================================================================
# Class_Ex3:
# Write Python script to find numbers between 1 to 3 in a given string.

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')


s1 = '1iu2hd1872y817239848921309'
def zero_or_more_b(text):
    import re
    pattern = re.compile(r'[1-3]')
    matches = pattern.finditer(text)
    for match in matches:
        print(match)

print(zero_or_more_b(s1))

print(20*'-' + 'End Q3' + 20*'-')
# =================================================================
# Class_Ex4:
# Write a Python script to find the a position of the substrings within a string.
# text = 'Python exercises, JAVA exercises, C exercises'
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q4' + 20*'-')
text = 'Python exercises, JAVA exercises, C exercises'
def find_substring(text, substring):
    import re
    pattern = re.compile(f'{substring}')
    matches = pattern.finditer(text)
    for match in matches:
        print(match)

print(find_substring(text, 'Python'))
print(find_substring(text, 'JAVA'))


print(20*'-' + 'End Q4' + 20*'-')
# =================================================================
# Class_Ex5:
# Write a Python script to find if two strings from a list starting with letter 'C'.
# words = ["Cython CHP", "Java JavaScript", "PERL S+"]
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q5' + 20*'-')
list = ["Cython CHP", "Java JavaScript", "PERL S+"]
def two_c(list):
    import re
    count = 0
    for string in list:
        pattern = re.compile(r'^C')
        matches = pattern.finditer(string)
        for match in matches:
            count += 1
    if count > 1:
        print('True')
    else:
        print('False')

print(two_c(list))


print(20*'-' + 'End Q5' + 20*'-')

# =================================================================
# Class_Ex6:
# Write a Python script to remove everything except chars and digits from a string.
# USe sub method
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q6' + 20*'-')
import re
string = '$$%$%32u34h9&*Y*&H9h3294hr&*&****'
cleaned = re.sub(r'[^a-zA-Z0-9]', '', string)
print(string)
print(cleaned)



print(20*'-' + 'End Q6' + 20*'-')
# =================================================================
# Class_Ex7:
# Scrape the the following website
# https://en.wikipedia.org/wiki/Natural_language_processing
# Find the tag which related to the text. Extract all the textual data.
# Tokenize the cleaned text file.
# print the len of the corpus and pint couple of the sentences.
# Calculate the words frequencies.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q7' + 20*'-')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from urllib import request
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from nltk import word_tokenize
from collections import Counter
html = request.urlopen('https://en.wikipedia.org/wiki/Natural_language_processing').read().decode('utf8')
raw = BeautifulSoup(html, 'html.parser').get_text()
tokens = word_tokenize(raw)
sents = sent_tokenize(raw)

print(len(tokens))
for i in range(3):
    print(sents[i])

counts = Counter(tokens)

for i in tokens:
    print(i, ':', counts[i]/len(tokens))

print(20*'-' + 'End Q7' + 20*'-')
# =================================================================
# Class_Ex8:
# Grab any text from Wikipedia and create a string of 3 sentences.
# Use that string and calculate the ngram of 1 from nltk package.
# Use BOW method and compare the most 3 common words.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q8' + 20*'-')
from nltk import ngrams
from nltk import word_tokenize
from nltk import FreqDist
string = ''
for i in sents[10:13]:
    string += i
tokens = word_tokenize(string)
onegram = FreqDist([gram for gram in ngrams(tokens, 1)]).most_common(3)
print(onegram)

print(20*'-' + 'End Q8' + 20*'-')
# =================================================================
# Class_Ex9:
# USe sklearn package and load 20 newsgroups dataset.
# Write a python script that accepts any string and do the following.
# 1- Tokenize the text
# 2- Doe word extraction and clean a text. USe regular expression to clean a text.
# 3- Generate BOW
# 4- Vectorize all the tokens.
# 5- The only package tou can use is numpy and re.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q9' + 20*'-')

from sklearn.datasets import fetch_20newsgroups
'''def function(text):'''








print(20*'-' + 'End Q9' + 20*'-')
# =================================================================
# Class_Ex10:
# Grab any text (almost a paragraph) from Wikipedia and call it text
# Preprocessing the text data (Normalize, remove special char, ...)
# Find total number of unique words
# Create an index for each word.
# Count number of the owrds.
# Define a function to calculate Term Frequency
# Define a function calculate Inverse Document Frequency
# Combining the TF-IDF functions
# Apply the TF-IDF Model to our text
# you are allowed to use just numpy and nltk tokenizer
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q10' + 20*'-')

text = "The eastern brown snake (Pseudonaja textilis) is a highly venomous snake of the family Elapidae, native to eastern and central Australia and southern New Guinea. Up to 2.4 metres (7.9 ft) long with a slender build, it has variable upperparts, pale brown to almost black, and a pale cream-yellow underside, often with orange or grey splotches. It was first described by André Marie Constant Duméril in 1854. The eastern brown snake is found in many habitats, though not in dense forests. It has become more common in farmland and on the outskirts of urban areas, preying mainly on the introduced house mouse. It is considered the world's second-most venomous land snake after the inland taipan, based on the toxicity of its venom in mice. According to one study, as a genus, brown snakes were responsible for 15 of 19 snakebite fatalities in Australia between 2005 and 2015."
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
stop_words = stopwords.words('english')
tokens = word_tokenize(text)
docs = sent_tokenize(text)
non_stopwords = [w for w in tokens if w not in stopwords.words('english')]
cleaned = [w.lower() for w in non_stopwords if w.isalnum()]
unique_words = len(set(cleaned))
print('Unique words: ', unique_words)
def tf(text, token):
    total_count = len(text)
    counts = Counter(text)
    tf = counts[token]/total_count
    return tf

def idf(docs, token):
    import numpy as np
    num_docs_containing = 1
    num_docs = len(docs)
    for doc in docs:
        if token in doc:
            num_docs_containing += 1
    idf = np.log(num_docs/num_docs_containing)
    return idf


for token in cleaned:
    for doc in docs:
        print(token, '// Doc #', docs.index(doc), ': ', tf(cleaned, str(token)) * idf(doc, str(token)))

print(20*'-' + 'End Q10' + 20*'-')
# =================================================================
# Class_Ex11:
# Grab arbitrary paragraph from any website.
# Creat  a list of stopwords manually.  Example :  stopwords = ['and', 'for', 'in', 'little', 'of', 'the', 'to']
# Create a list of ignore char Example: ' :,",! '
# Write a LSA class with the following functions.
# Parse function which tokenize the words lower cases them and count them. Use dictionary; keys are the tokens and value is count.
# Clac function that calculate SVD.
# TFIDF function
# Print function which print out the TFIDF matrix, first 3 columns of the U matrix an irst 3 rows of the Vt matrix
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q11' + 20*'-')

text = "The eastern brown snake (Pseudonaja textilis) is a highly venomous snake of the family Elapidae, native to eastern and central Australia and southern New Guinea. Up to 2.4 metres (7.9 ft) long with a slender build, it has variable upperparts, pale brown to almost black, and a pale cream-yellow underside, often with orange or grey splotches. It was first described by André Marie Constant Duméril in 1854. The eastern brown snake is found in many habitats, though not in dense forests. It has become more common in farmland and on the outskirts of urban areas, preying mainly on the introduced house mouse. It is considered the world's second-most venomous land snake after the inland taipan, based on the toxicity of its venom in mice. According to one study, as a genus, brown snakes were responsible for 15 of 19 snakebite fatalities in Australia between 2005 and 2015."

stopwords = ['and', 'for', 'in', 'little', 'of', 'the', 'to',]
ignore_chars = ['!','@','#,''$','%','^','&','*','(',')','_','-','+','=',':',';',',','.','<','>','/','?','`','~','|','"']
'''
class LSA:
    def __init__(self):
    def tokenize(self):
        stopwords = ['and', 'for', 'in', 'little', 'of', 'the', 'to', ]
        ignore_chars = ['!', '@', '#,''$', '%', '^', '&', '*', '(', ')', '_', '-', '+', '=', ':', ';', ',', '.', '<', '>', '/', '?', '`', '~', '|', '"']
        tokens = [token for token in ]

'''

print(20*'-' + 'End Q11' + 20*'-')
# =================================================================
# Class_Ex12:
# Use the following doc
# doc = ["An intern at OpenAI", "Developer at OpenAI", "A ML intern", "A ML engineer" ]
# Calculate the binary BOW.
# Use LSA method and distinguish two different topic from the document. Sent 1,2 is about OpenAI and sent3, 4 is about ML.
# Use pandas to show the values of dataframe and lsa components. Show there is two distinct topic.
# Use numpy take the absolute value of the lsa matrix sort them and use some threshold and see what words are the most important.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q12' + 20*'-')







print(20*'-' + 'End Q12' + 20*'-')








