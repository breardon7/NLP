#-------------------Load dataset & tokenize (word and sents)------------------------------
from nltk import word_tokenize
file = open('data.txt').read()
tokens = word_tokenize(file)
print(tokens[0:10])

# -------------- find numbers and any words has numbers in it--------------------------------------
numbs = [word for word in file if word.isnumeric()]
print('numbers: ', numbs)
word_with_numbs = [s for s in tokens if any(c.isdigit() for c in s) and any(c.isnumeric() for c in s)]
print('words with numbers:', word_with_numbs)

# --------------remove punctuations--------------------------------------
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
cleaned = tokenizer.tokenize(file)
print('cleaned: ', cleaned)

# --------------Clean and Save Data--------------------------------------
new_file = open('Cleaned_File.txt', 'w')
for i in range(len(cleaned)):
    new_file.write(cleaned[i] + " ")

# ----------------Normalize Text and Remove Stop Words----------------------
from nltk.corpus import stopwords
normal = [word for word in tokens if word.isalpha()]
normal = [word.lower() for word in normal]
normal = [word for word in normal if not word in stopwords.words('english')]
print('normal: ', normal)

# -------------------------Most common words and once happened word------------------------------------------------
from nltk import FreqDist
most = FreqDist(normal).most_common(10)
print('most: ', most)
once = [k for (k,v) in FreqDist(normal).items() if v == 1]
print('once: ', once)


# -------------------------Words more that 15 chars------------------------------------------------
more_than_15 = [word for word in tokens if len(word) > 15]
print(more_than_15)

# -------------------------Find the phonemes of word mine------------------------------------------------
from nltk.corpus import cmudict
arpabet = cmudict.dict()
phenomes = arpabet['mine']
print('phenomes: ', phenomes)



# -------------------------Find miss spelled words------------------------------------------------
from nltk.corpus import words
vocab = set(words.words())
misspelled = [word for word in normal if word not in vocab]
print('misspelled: ', misspelled)



# -------------------------What this text is about------------------------------------------------
# This text is about asking people to join a group where they talk about random subjects.
# I cam to this conclusion by looking at the words that most frequently occur after normalizing
# and removing stop words from the tokens list.





