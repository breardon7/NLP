# ====================================Part 1================================
print(20*'-' + 'Part1' + 20*'-')

import nltk
import numpy as np

BOW = ['i', 'am', 'excited', 'about', 'data', 'real', 'with', 'classifier']

word2count = {}
for word in BOW:
    if word not in word2count.keys():
        word2count[word] = 1
    else:
        word2count[word] += 1
print(word2count)
# ====================================Part 2================================
print(20*'-' + 'Part2' + 20*'-')


sent1 = 'I am excited about the neural network.'
sent2 = 'We will not test the classifier with real data.'
inputs = np.array(['I am excited about the neural network.', 'We will not test the classifier with real data.'])
targets = np.array([1,0])




# ====================================Part 3================================
print(20*'-' + 'Part3' + 20*'-')







# ====================================Part 4================================
print(20*'-' + 'Part4' + 20*'-')






