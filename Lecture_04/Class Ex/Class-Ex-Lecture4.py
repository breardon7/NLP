# =================================================================
# Class_Ex1:
# Write a function that checks a string contains only a certain set of characters
# (all chars lower and upper case with all digits).
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')
text = 'This is a test string 123^&#@*98'

def all(text):
    import re
    pattern = re.compile(r'\w')
    matches = pattern.finditer(text)
    for match in matches:
        print(match)

print(all(text))




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









print(20*'-' + 'End Q6' + 20*'-')
# =================================================================
# Class_Ex7:
# Scrape the the following website
# https://en.wikipedia.org/wiki/Natural_language_processing
# Find the tag which related to the text. Extract all the textual data.
# Tokenize the cleaned text file.
# print the len of the corpus and pint couple of th esenetences.
# Calculate the words frequencies.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q7' + 20*'-')









print(20*'-' + 'End Q7' + 20*'-')
# =================================================================
# Class_Ex8:
# Grab any text from Wikipedia and create a string of 3 sentences.
# Use that string and calculate the ngram of 1 from nltk package.
# Use BOW method and compare the most 3 common owrds.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q8' + 20*'-')











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
# =================================================================
# Class_Ex13:
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q13' + 20*'-')









print(20*'-' + 'End Q13' + 20*'-')
# =================================================================
# Class_Ex14:
#

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q14' + 20*'-')









print(20*'-' + 'End Q14' + 20*'-')

# =================================================================









