# =================================================================
# Class_Ex1:
# Use NLTK Book fnd which the related Sense and Sensibility.
# Produce a dispersion plot of the four main protagonists in Sense and Sensibility:
# Elinor, Marianne, Edward, and Willoughby. What can you observe about the different
# roles played by the males and females in this novel? Can you identify the couples?
# Explain the result of plot in a couple of sentences.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')
from nltk.book import text2
text2.dispersion_plot(["Elinor", "Marianne", "Edward", "Willoughby"])

print(20*'-' + 'End Q1' + 20*'-')

# =================================================================
# Class_Ex2:
# What is the difference between the following two lines of code? Explain in details why?
# Make up and example base don your explanation.
# Which one will give a larger value? Will this be the case for other texts?
# 1- sorted(set(w.lower() for w in text1))
# 2- sorted(w.lower() for w in set(text1))
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')

# Code 1 normalizes the text prior to applying the set() function,
# whereas code 2 does not. This means code 2 will create a set that might
# contain distinct elements that only differ in lower/upper cases
# (i.e. 'Hello' and 'hello' will both exist in the set), but code 1 will
# prevent that by normalizing the case of each element prior to creating
# a set. Code 2 will be larger and will most likely happen with other
# texts when using these two lines of code.


print(20*'-' + 'End Q2' + 20*'-')
# =================================================================
# Class_Ex3:
# Find all the four-letter words in the Chat Corpus (text5).
# With the help of a frequency distribution (FreqDist), show these words in decreasing order of frequency.
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')
from nltk.book import text5
from nltk import FreqDist
four = [w for w in text5 if len(w) == 4]
most = FreqDist(four).most_common(len(four))
print(most)

print(20*'-' + 'End Q3' + 20*'-')
# =================================================================
# Class_Ex4:
# Write expressions for finding all words in text6 that meet the conditions listed below.
# The result should be in the form of a list of words: ['word1', 'word2', ...].
# a. Ending in ise
# b. Containing the letter z
# c. Containing the sequence of letters pt
# d. Having all lowercase letters except for an initial capital (i.e., titlecase)
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q4' + 20*'-')
from nltk.book import text6
ex4 = [w for w in text6 if w.endswith('ise') and 'z' in w and 'pt' in w and w.title()]
print(ex4)

print(20*'-' + 'End Q4' + 20*'-')
# =================================================================
# Class_Ex5:
#  Read in the texts of the State of the Union addresses, using the state_union corpus reader.
#  Count occurrences of men, women, and people in each document.
#  What has happened to the usage of these words over time?
# Since there would be a lot of document use every couple of years.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q5' + 20*'-')

from nltk.corpus import state_union
from nltk import ConditionalFreqDist
cfd = ConditionalFreqDist((words, f[4:10]) for f in state_union.fileids() for w in state_union.words(f) for words in ['men', 'women', 'people'] if w.lower().startswith(words))
cfd.plot()

print(20*'-' + 'End Q5' + 20*'-')

# =================================================================
# Class_Ex6:
# The CMU Pronouncing Dictionary contains multiple pronunciations for certain words.
# How many distinct words does it contain? What fraction of words in this dictionary have more than one possible pronunciation?
#
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q6' + 20*'-')
from nltk.corpus import cmudict
from nltk import FreqDist
entries = cmudict.entries()
words = [w for w, p in entries]
distinct = len(set(words))
multi = len([k for (k,v) in FreqDist(words).items() if v > 1])/distinct
print(distinct)
print(multi)


print(20*'-' + 'End Q6' + 20*'-')
# =================================================================
# Class_Ex7:
# What percentage of noun synsets have no hyponyms?
# You can get all noun synsets using wn.all_synsets('n')
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q7' + 20*'-')
from nltk.corpus import wordnet as wn
nouns = [n for n in wn.all_synsets('n') if len(n.hyponyms())==0]
all_nouns = [n for n in wn.all_synsets('n')]
print(len(nouns)/len(all_nouns))



print(20*'-' + 'End Q7' + 20*'-')
# =================================================================
# Class_Ex8:
# Write a program to find all words that occur at least three times in the Brown Corpus.
# USe at least 2 different method.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q8' + 20*'-')
from nltk.corpus import brown
from collections import Counter
three = [k for (k,v) in FreqDist(brown.words()).items() if v > 2]
#print(three)
three = [w for w in set(brown.words()) if FreqDist(brown.words())[w]>2]
#print(three)


print(20*'-' + 'End Q8' + 20*'-')
# =================================================================
# Class_Ex9:
# Write a function that finds the 50 most frequently occurring words of a text that are not stopwords.
# Test it on Brown corpus (humor), Gutenberg (whitman-leaves.txt).
# Did you find any strange word in the list? If yes investigate the cause?
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q9' + 20*'-')
from nltk.corpus import gutenberg
def most_freq_non_stop_words(text):
    from nltk.corpus import stopwords
    non_stopwords = [w for w in text if w not in stopwords.words('english')]
    cleaned = [w.lower() for w in non_stopwords if w.isalnum()]
    most_frequent = FreqDist(cleaned).most_common(50)
    return most_frequent

print(most_freq_non_stop_words(brown.words(categories='humor')))
print(most_freq_non_stop_words(gutenberg.words('whitman-leaves.txt')))
# I did not find any strange words in the text
print(20*'-' + 'End Q9' + 20*'-')
# =================================================================
# Class_Ex10:
# Write a program to create a table of word frequencies by genre, like the one given in 1 for modals.
# Choose your own words and try to find words whose presence (or absence) is typical of a genre. Discuss your findings.

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q10' + 20*'-')
cfd = ConditionalFreqDist((genre,word)for genre in brown.categories() for word in brown.words(categories=genre))
words = ['school', 'fun', 'hello', 'mom', 'dad', 'electric', 'sky']
cfd.tabulate(conditions=brown.categories(), samples=words)

print(20*'-' + 'End Q10' + 20*'-')
# =================================================================
# Class_Ex11:
#  Write a utility function that takes a URL as its argument, and returns the contents of the URL,
#  with all HTML markup removed. Use from urllib import request and
#  then request.urlopen('http://nltk.org/').read().decode('utf8') to access the contents of the URL.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q11' + 20*'-')


def utility(url):
    from urllib import request
    from bs4 import BeautifulSoup
    from nltk import word_tokenize
    html = request.urlopen(url).read().decode('utf8')
    raw = BeautifulSoup(html, 'html.parser').get_text()
    tokens = word_tokenize(raw)
    return tokens
print(utility('https://en.wikipedia.org/wiki/Data_science'))

print(20*'-' + 'End Q11' + 20*'-')
# =================================================================
# Class_Ex12:
# Read in some text from a corpus, tokenize it, and print the list of all
# wh-word types that occur. (wh-words in English are used in questions,
# relative clauses and exclamations: who, which, what, and so on.)
# Print them in order. Are any words duplicated in this list,
# because of the presence of case distinctions or punctuation?
# Note Use: Gutenberg('bryant-stories.txt')
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q12' + 20*'-')
from nltk import word_tokenize
text = gutenberg.words('bryant-stories.txt')
wh_words = ['who', 'what', 'where', 'when', 'why', 'which', 'whom', 'whose']
wh_list = [w for w in text if w.lower() in wh_words]
print(wh_list)

# There are duplicates due to case distinctions

print(20*'-' + 'End Q12' + 20*'-')
# =================================================================
# Class_Ex13:
# Write code to access a  webpage and extract some text from it.
# For example, access a weather site and extract  a feels like temprature..
# Note use the following site https://darksky.net/forecast/40.7127,-74.0059/us12/en
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q13' + 20*'-')
from urllib import request
from bs4 import BeautifulSoup
from nltk import sent_tokenize
html = request.urlopen('darksky.net//forecast//40.7127,-74.0059//us12//en').read().decode('utf8')
raw = BeautifulSoup(html, 'html.parser').get_text()
text = sent_tokenize(raw)
temp = [s for s in text if 'temperature' in s]
print(temp)


print(20*'-' + 'End Q13' + 20*'-')
# =================================================================
# Class_Ex14:
# Train a bigram tagger with no backoff tagger, and run it on some of the training data.
# Next, run it on some new data. What happens to the performance of the tagger? Why?
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q14' + 20*'-')

from nltk import BigramTagger
from nltk.corpus import treebank
tagged = treebank.tagged_sents()
ln =  int(0.9*len(tagged))
train = tagged[:ln]
test = tagged[ln:]
bgt = BigramTagger(train=train)
print('test results 2: ', bgt.evaluate(test))
print('train results 2: ', bgt.evaluate(train))

# The test results are much worse than the training data results.
# This is most likely due to the tagger overfitting on the train data.

print(20*'-' + 'End Q14' + 20*'-')

# =================================================================
# Class_Ex15:
# Use sorted() and set() to get a sorted list of tags used in the Brown corpus, removing duplicates.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q15' + 20*'-')

print(sorted(set(brown.words())))


print(20*'-' + 'End Q15' + 20*'-')

# =================================================================
# Class_Ex16:
# Write programs to process the Brown Corpus and find answers to the following questions:
# 1- Which nouns are more common in their plural form, rather than their singular form? (Only consider regular plurals, formed with the -s suffix.)
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q16' + 20*'-')
cfd = ConditionalFreqDist(brown.tagged_words())
conditions = cfd.conditions()
for condition in conditions:
	if cfd[condition]['NNS'] > cfd[condition]['NN']:
		print(condition)


print(20*'-' + 'End Q16' + 20*'-')


