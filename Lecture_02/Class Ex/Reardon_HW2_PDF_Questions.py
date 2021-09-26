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
#print([wnl.lemmatize(t) for t in tokens])
# v.
# vi.
whale = tokens.count('whale') + tokens.count('Whale')
print('whale count: ', whale/len(tokens)*100)
# vii.
from nltk import FreqDist
freq_20 = FreqDist(tokens).most_common(20)
print(freq_20)
# viii.
# ix.
# x.
# xi.
# xii.

# =================================================================
# E.2
# ----------------------------------------------------------------


# =================================================================
# E.3
# ----------------------------------------------------------------


# =================================================================
# E.4
# ----------------------------------------------------------------