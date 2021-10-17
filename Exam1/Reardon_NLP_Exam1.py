# import packages
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# read in data
df = pd.read_csv('data.csv')
df['cleaned_text'] = ''
vc = TfidfVectorizer()
cv = CountVectorizer(analyzer='word', ngram_range=(2, 2))
ps = PorterStemmer()
lm = WordNetLemmatizer()

# preprocess text data
for i in range(len(df['text'])):
    tokens = word_tokenize(df['text'][i])
    lower = [token.lower() for token in tokens if token.isalnum()]
    no_stop_words = [token for token in lower if token not in stopwords.words('english')]
    stemmed = [ps.stem(token) for token in no_stop_words]
    #lemmed = [lm.lemmatize(token) for token in no_stop_words]
    cleaned_string = ' '.join(stemmed)
    df['cleaned_text'][i] = cleaned_string

# Vectorize text data
corpus = [row for row in df['cleaned_text']]
vectorized_data = vc.fit_transform(corpus)
#vectorized_data = cv.fit_transform(corpus)
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
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))

# Exam code
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)

df_test['labels'] = clf.predict(insert df_test vectorized data)'''
#df_test.to_csv(r'C:\Users\brear\OneDrive\Desktop\NLP\Exam1\HW5_Test_Output')