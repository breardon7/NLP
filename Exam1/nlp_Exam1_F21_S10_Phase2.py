#============================Problem Set 1=======================================================================================================
# This dataset is about review and feedback of customers about the clothing.
# There are multiple features such as
# Clothing ID	Age	Title	Review Text	Rating	Recommended IND	Positive Feedback Count	Division Name	Department Name	Class Name
# Lets work on this set of data and answer the following questions.
#**********************************




#**********************************
# Q1:
# Loading Dataset and get some statistics about the features.
import pandas as pd
df = pd.read_csv('Data.csv')
print(df.columns)
print(df.head())
print(df.dtypes)
print(df.describe(percentiles=[.25, .5, .75]))


#**********************************
# Q2:
# Find a most popular clothing ID and created a new dataframe.
# Next extract all the reviews from the new dataframe.

print(df['Clothing ID'].mode())
df1 = df[df['Clothing ID'] == 1078]
df1['Review Text'].dropna(inplace=True)
reviews = df1['Review Text'].astype(str)
text = ''.join(reviews)
#**********************************
# Q3:
# Tokenize the new corpus with nltk.
# Remove All special chars and symbols using regular expression and lower cases all the text.
# How many chars is removed.
# Remove stop words and use the nltk stopwords list.
# print the length of the filtered tokenized text.

from nltk import word_tokenize
from nltk.corpus import stopwords
import re
tokens = word_tokenize(text)
for token in tokens:
    token = re.sub('[^A-Za-z0-9]', '', token)
tokens = [token.lower() for token in tokens if token not in stopwords.words('english')]
print(len(tokens))



#**********************************
# Q4:
# Normalize text to its root use nltk package.

from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
lemmed = [lm.lemmatize(token) for token in tokens]
print(lemmed)
#**********************************
# Q5:
# Used the stemmed tokenized set and calculate the TFIDF
# used the LSA method and find try to categorize the terms into 2 context. Just use 10 terms in LSA (restrict LSA to use the 10 terms).

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(lemmed)

from sklearn.decomposition import TruncatedSVD
lsa = TruncatedSVD(n_components=2,n_iter=100)
lsa.fit(X)



