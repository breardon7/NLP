# ====================================Part 1================================
print(20*'-' + 'Part1' + 20*'-')

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
import matplotlib.pyplot as plt

sent1 = 'I am excited about the neural network.'
sent2 = 'We will not test the classifier with real data.'
inputs = np.array(['I am excited about the neural network.', 'We will not test the classifier with real data.'])

BOW = ['I', 'am', 'excited', 'about', 'data', 'real', 'with', 'classifier']

my_dict = {}
for word in BOW:
    if word not in my_dict.keys():
        my_dict[word] = 1
    else:
        my_dict[word] += 1
print(my_dict)


X = []
for row in inputs:
    tokens = word_tokenize(row)
    vector = []
    for word in BOW:
        if word in tokens:
            vector.append(1)
        else:
            vector.append(0)
    X.append(np.array(vector))
print(X)

# ====================================Part 2================================
print(20*'-' + 'Part2' + 20*'-')

inputs = X
targets = np.array([1,0])

# ====================================Part 3================================
print(20*'-' + 'Part3' + 20*'-')

def hardlim(x):
    if x < 0:
        return 0
    else:
        return 1

def ppn(p,t):
    epoch = 50
    inputs = 8 # classes
    b = np.ones(1)
    w = np.ones(inputs)
    lr = 0.15
    epoch_iter = 1
    epoch_values = []
    E_values = []
    for j in range(epoch):
        E = 0.0
        for i in range(len(p)):
            n = np.dot(w,p[i]) + b
            a = hardlim(n)
            e = t[i] - a
            w = w + lr*e*p[i]
            b = b*lr + e
            E += e * e
        E_values.append(E)
        epoch_values.append(epoch_iter)
        epoch_iter += 1

    print(w)
    #E.append(e)
    plt.plot(epoch_values, E_values)
    plt.ylabel('E')
    plt.xlabel('epoch')
    plt.title('Error vs Epoch')
    plt.show()
    for i in range(len(p)):
        plt.plot(p[i][0], p[i][1], 'rx' if (t[i] == 1) else 'bx')
    x = np.linspace(0, 5)
    m = -w[0]/w[1]
    y = m*x + b
    plt.plot(x,y)
    plt.arrow(2.4, 3, w[0], w[1], head_width = .2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Hardlim')
    plt.show()





# ====================================Part 4================================
print(20*'-' + 'Part4' + 20*'-')



ppn(inputs, targets)


