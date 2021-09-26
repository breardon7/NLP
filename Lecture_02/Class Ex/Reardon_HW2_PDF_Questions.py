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
# i. & ii.
def scrape_raw(url):
    from urllib import request
    from bs4 import BeautifulSoup
    html = request.urlopen(url).read().decode('utf8')
    raw = BeautifulSoup(html, 'html.parser').get_text()
    return raw
url = 'https://en.wikipedia.org/wiki/Benjamin_Franklin'
wiki_text = [scrape_raw(url)]
# iii.
def unknown(text):
    from nltk.corpus import words
    for word in text:
        if word in words.words():
            text.remove(word)
    return text
#print(unknown(wiki_text))
# iv.
novel_words = word_tokenize(open('twitter.txt', encoding='utf-8').read())
# v
from nltk import PorterStemmer
ps = PorterStemmer()
stemmed = [ps.stem(w) for w in novel_words]
novel_stems = unknown(stemmed)
print(novel_stems)
# vi
tagged_names = pos_tag(novel_stems)
proper_names = [word for word,pos in tagged_names if pos == 'NNP']
print(proper_names)
# =================================================================
# E.3
# ----------------------------------------------------------------
# i.
twitter = open('twitter.txt', 'r', encoding='utf-8')
print(twitter.readline())
print(twitter.readline())
print(twitter.readline())
# ii.
twit_sent = twitter.read().split(sep='\n')
# iii.
twitter1 = open('twitter.txt', encoding='utf-8').read()
tokens = word_tokenize(twitter1)
# iv.
train = twit_sent[:40000]
test = twit_sent[40000:]
print(len(twit_sent))
# v.
freq = FreqDist(tokens).most_common(len(tokens))
print(freq)