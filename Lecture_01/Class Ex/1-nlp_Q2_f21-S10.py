#-------------------Load dataset & tokenize (word and sents)------------------------------
from nltk import word_tokenize
file = open('data.txt').read()
tokens = word_tokenize(file)
print(tokens[0:10])



# -------------- find numbers and any words has numbers in it--------------------------------------
numbs = []
for i in file:
    if i.isnumeric():
        numbs += i
print(numbs)
word_with_numbs = []
word_with_numbs.append(s for s in tokens if not any(c.isdigit() for c in s))
'''for i in range(len(tokens)):
    if tokens[i].isalnum():
        word_with_numbs += tokens[i]'''
print(word_with_numbs)




# --------------remove punctuations--------------------------------------

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
cleaned = tokenizer.tokenize(file)
print(cleaned)


# --------------Clean and Save Data--------------------------------------
new_file = open('Cleaned_File.txt', 'w')
for i in range(len(cleaned)):
    new_file.write(cleaned[i] + " ")



# ----------------Normalize Text and Remove Stop Words----------------------






# -------------------------Most common words and once happened word------------------------------------------------






# -------------------------Words more that 15 chars------------------------------------------------





# -------------------------Find the phonemes of word mine------------------------------------------------





# -------------------------Find miss spelled words------------------------------------------------







# -------------------------What this text is about------------------------------------------------






