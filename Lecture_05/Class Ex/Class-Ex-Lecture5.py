# =================================================================
# Class_Ex1:
#  Use the following datframe as the sample data.
# Find the conditional probability of Char given the Occurrence.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
print(df.columns)
# P(Char|Occ) = (P(Occ|Char)*P(Char))/P(Occ)








print(20*'-' + 'End Q1' + 20*'-')

# =================================================================
# Class_Ex2:
# Use the following dataframe as the sample data.
# Find the conditional probability occurrence of thw word given a sentiment.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')












print(20*'-' + 'End Q2' + 20*'-')
# =================================================================
# Class_Ex3:
# Read the data.csv file.
# Answer the following question
# 1- In this dataset we have a lot of responses in text and each response has a label.
# 2- Our goal is to correctly model the texts into its label.
# Hint: you need to read the text responses and perform preprocessing on it.
# such as normalization, legitimation, cleaning, stopwords removal and POS tagging.
# then use any methods you learned in the lecture to convert each response into meaningful numbers.
# 3- Apply Naive bayes and look at appropriate evaluation metric.
# 4- Explain your results very carefully.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')
'''# import packages
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# read in data
df = pd.read_csv('data.csv')
df['cleaned_text'] = ''
vc = TfidfVectorizer()
ps = PorterStemmer()

# preprocess text data
for i in range(len(df['text'])):
    tokens = word_tokenize(df['text'][i])
    lower = [token.lower() for token in tokens if token.isalnum()]
    no_stop_words = [token for token in lower if token not in stopwords.words('english')]
    stemmed = [ps.stem(token) for token in no_stop_words]
    cleaned_string = ' '.join(stemmed)
    df['cleaned_text'][i] = cleaned_string

# Vectorize text data
corpus = [row for row in df['cleaned_text']]
vectorized_data = vc.fit_transform(corpus)
#print(vectorized_data.shape)

# Create train set
X = vectorized_data
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = MultinomialNB(alpha=0.5)
clf.fit(X_train,y_train)

# Test on test set
predicted = clf.predict(X_test)

# Classification report
print('----------------Naive Bayes----------------')
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))
'''
# Results
#Given the F1 scores of 0.83 and 0.80 for lable 1 and label 2 respectively,
# we can assume that the model will classify unseen data fairly well. The model
# causes more false positives for label 1 than label 2 given that label 1 has
# a lower precision, but causes more false negatives for label 2 given that label
# 2 has a lower recall than label 1.


print(20*'-' + 'End Q3' + 20*'-')
# =================================================================
# Class_Ex4:
# Use Naive bayes classifier for this problem,
# Write a text classification pipeline to classify movie reviews as either positive or negative.
# Find a good set of parameters using grid search. hint: grid search on n gram
# Evaluate the performance on a held out test set.
# hint1: use nltk movie reviews dataset
# from nltk.corpus import movie_reviews

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q4' + 20*'-')












print(20*'-' + 'End Q4' + 20*'-')
# =================================================================
# Class_Ex5:
# Calculate accuracy percentage between two lists
# calculate a confusion matrix
# Write your own code - No packages
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q5' + 20*'-')

list1 = [1,2,3,4,5]
list2 = [1,2,3,5,4]
acc_perc = sum(1 for x,y in zip(list1,list2) if x == y) / len(list1)
print('Accuracy Percentage: ', acc_perc)

'''
# create confusion matrix
'''


print(20*'-' + 'End Q5' + 20*'-')
# =================================================================
# Class_Ex6:
# Read the data.csv file.
# Answer the following question
# 1- In this dataset we have a lot of responses in text and each response has a label.
# 2- Our goal is to correctly model the texts into its label.
# Hint: you need to read the text responses and perform preprocessing on it.
# such as normalization, legitimation, cleaning, stopwords removal and POS tagging.
# then use any methods you learned in the lecture to convert each response into meaningful numbers.
# 3- Apply Logistic Regression  and look at appropriate evaluation metric.
# 4- Apply LSA method and compare results.
# 5- Explain your results very carefully.

# ----------------------------------------------------------------
print(20*'-' + 'Begin Q6' + 20*'-')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn import metrics

# read in data
df = pd.read_csv('data.csv')
df['cleaned_text'] = ''
vc = TfidfVectorizer()
ps = PorterStemmer()

# preprocess text data
for i in range(len(df['text'])):
    tokens = word_tokenize(df['text'][i])
    lower = [token.lower() for token in tokens if token.isalnum()]
    no_stop_words = [token for token in lower if token not in stopwords.words('english')]
    stemmed = [ps.stem(token) for token in no_stop_words]
    cleaned_string = ' '.join(stemmed)
    df['cleaned_text'][i] = cleaned_string

# Vectorize text data
corpus = [row for row in df['cleaned_text']]
vectorized_data = vc.fit_transform(corpus)
#print(vectorized_data.shape)

# Create train set
X = vectorized_data
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = LogisticRegression()
clf.fit(X_train,y_train)

# Test on test set
predicted = clf.predict(X_test)

# Classification report
print('----------------Logistic Regression----------------')
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))

'''
lsa = TruncatedSVD(n_components=2,n_iter=100)
lsa.fit(X_train,y_train)

# Test on test set
predicted = lsa.predict(X_test)

# Classification report
print('----------------LSA----------------')
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))
'''











print(20*'-' + 'End Q6' + 20*'-')

# =================================================================
# Class_Ex7:
# Use logistic regression classifier for this problem,
# Write a text classification pipeline to classify movie reviews as either positive or negative.
# Find a good set of parameters using grid search. hint: grid search on n gram
# Evaluate the performance on a held out test set.
# hint1: use nltk movie reviews dataset
# from nltk.corpus import movie_reviews
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q7' + 20*'-')



