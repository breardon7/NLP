# =================================================================
# Class_Ex1:
#  Use the following datframe as the sample data.
# Find the conditional probability of Char given the Occurrence.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')

import pandas as pd
import numpy as np
df = pd.DataFrame({'Char':['f', 'b', 'f', 'b','f', 'b', 'f', 'f'], 'Occurance':['o1', 'o1', 'o2', 'o3','o2', 'o2', 'o1', 'o3'], 'C':np.random.randn(8), 'D':np.random.randn(8)})

# P(Char|Occ) = (P(Occ|Char)*P(Char))/P(Occ)
cond_prob = df.groupby(['Char']).Occurance.value_counts()/df.groupby(['Char']).Occurance.count()
print(cond_prob)

print(20*'-' + 'End Q1' + 20*'-')

# =================================================================
# Class_Ex2:
# Use the following dataframe as the sample data.
# Find the conditional probability occurrence of thw word given a sentiment.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')

df1 = pd.DataFrame({'Word': ['Good', 'Bad', 'Awesome', 'Beautiful', 'Terrible', 'Horrible'],
                     'Occurrence': ['One', 'Two', 'One', 'Three', 'One', 'Two'],
                     'sentiment': ['P', 'N', 'P', 'P', 'N', 'N'],})

cond_prob = df1.groupby(['Word']).sentiment.value_counts()/df1.groupby(['Word']).sentiment.count()
print(cond_prob)


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
# import packages
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

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

# Train model
clf = MultinomialNB()
clf.fit(X_train,y_train)

# Test on test set
predicted = clf.predict(X_test)

# Classification report
print('----------------Naive Bayes----------------')
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))

# Results
# Given the F1 scores of 0.83 and 0.80 for label 1 and label 2 respectively,
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

from nltk.corpus import movie_reviews
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

files = list(movie_reviews.fileids()[:4000])

text = []
label = []
for i in range(len(files)):
    text.append(movie_reviews.raw(fileids=files[i]))
for i in files:
    label.append(i[:3])

df = pd.DataFrame({'text': text, 'label': label})
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

# Create train set
X = vectorized_data
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
parameter_space = {
    'alpha': [1,.5,.2],
    'fit_prior': ['True', 'False'],
}

mnb = MultinomialNB()
clf = GridSearchCV(mnb, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train,y_train)

# Test on test set
predicted = clf.predict(X_test)

# Classification report
print('----------------Naive Bayes Movie Review----------------')
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))

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


def confusion_matrix(act, pred):
    classes = np.unique(act)
    conf_mat = np.zeros((len(classes), len(classes)), dtype=int)
    for i in range(len(classes)):
        for j in range(len(classes)):
           conf_mat[i, j] = np.sum((act == classes[i]) & (pred == classes[j]))
           print(np.sum((act == classes[i]) & (pred == classes[j])))
    return conf_mat


actual = [2,2,2,1,0,0,1,2,2,1]
predicted = [1,0,2,1,0,1,1,0,2,1]

print(confusion_matrix(actual, predicted))



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

from nltk.corpus import movie_reviews
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

files = list(movie_reviews.fileids()[:4000])

text = []
label = []
for i in range(len(files)):
    text.append(movie_reviews.raw(fileids=files[i]))
for i in files:
    label.append(i[:3])

df = pd.DataFrame({'text': text, 'label': label})
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

# Create train set
X = vectorized_data
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Fit model
parameter_space = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'fit_intercept': ['True', 'False'],
    'dual': ['True', 'False'],
}
lr = LogisticRegression()
clf = GridSearchCV(lr, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train,y_train)

# Test on test set
predicted = clf.predict(X_test)

# Classification report
print('----------------Logistic Regression Movie Review----------------')
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))

