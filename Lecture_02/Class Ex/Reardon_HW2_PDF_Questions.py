# =================================================================
# E.1
# ----------------------------------------------------------------
# i.
from nltk.book import text1
from nltk import word_tokenize
tokens = [w for w in text1]
# ii.
print('token count: ', len(tokens))
# iii.
print('unique token count: ', len(set(tokens)))
# iv
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
lemmatized = [wnl.lemmatize(w, 'v') for w in tokens]
print(len(set(lemmatized)))
# v.
print('lexical diversity: ', len(tokens)/len(set(tokens)))
# vi.
whale = tokens.count('whale') + tokens.count('Whale')
print('whale count: ', whale/len(tokens)*100)
# vii.
from nltk import FreqDist
freq_20 = FreqDist(tokens).most_common(20)
print(freq_20)
# viii.
gr_six = [w for w in tokens if len(w) > 6]
freq_160 = [k for (k,v) in FreqDist(gr_six).items() if v > 160]
print(freq_160)
# ix.
longest = max(tokens, key=len)
print(longest, len(longest))
# x.
freq_2000 = [(k,v) for (k,v) in FreqDist(gr_six).items() if v > 2000]
print(freq_2000)
# no words have a frequency of > 2000.
# xi.
from nltk import sent_tokenize
raw_moby = open('moby.txt', 'r').read()
sentences = sent_tokenize(raw_moby)
avg = sum(map(len, sentences))/len(sentences)
print('average token length per sentence: ', avg)

# xii.
from nltk import pos_tag
from collections import Counter
import itertools
tags = pos_tag(tokens)
counts = Counter(tag for word, tag in tags)
print('5 most common tags: ', counts.most_common(5))

# =================================================================
# E.2
# ----------------------------------------------------------------


# =================================================================
# E.3
# ----------------------------------------------------------------

