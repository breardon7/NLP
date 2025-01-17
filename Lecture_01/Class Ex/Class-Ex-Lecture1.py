# =================================================================
# Class_Ex1:
# Write a function that prints all the chars from string1 that appears in string2.
# Note: Just use the Strings functionality no packages should be used.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')

s1 = 'dfgdfhd krifk39'
s2 = ' ierfmo 3458j ijer0g'
def s1s2(x,y):
    z = []
    for i in x:
        if i in y:
            print(i)

print(s1s2(s1,s2))


print(20*'-' + 'End Q1' + 20*'-')

# =================================================================
# Class_Ex2:
# Write a function that counts the numbers of a particular letter in a string.
# For example count the number of letter a in abstract.
# Note: Compare your function with a count method
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')


def count_num(string):
    count = 0
    for i in string:
        count += 1
    return count

print(count_num(s1))

print(20*'-' + 'End Q2' + 20*'-')
# =================================================================
# Class_Ex3:
# Write a function that reads the Story text and finds the strings in the curly brackets.
# Note: You are allowed to use the strings methods
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')
variable = 'nice job!'
text = 'example string {variable} brackets'
def story_text(text):
    '''return text.format()'''

    start = text.find('{')
    end = text.find('}')
    return text[start+1:end]


print(story_text(text))

print(20*'-' + 'End Q3' + 20*'-')
# =================================================================
# Class_Ex4:
# Write a function that read the first n lines of a file.
# Use test_1.txt as sample text.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q4' + 20*'-')

text = open('test_1.txt', 'r', encoding='utf-8')
def readNlines(n, text):
    for i in range(n+1):
        return text.readline()

print(readNlines(5,text))


print(20*'-' + 'End Q4' + 20*'-')
# =================================================================
# Class_Ex5:
# Write a function that read a file line by line and store it into a list.
# Use test_1.txt as sample text.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q5' + 20*'-')

text = open("test_1.txt", 'r', encoding='utf-8')

def linebyline(text):
    return list(text.readlines())
print(linebyline(text))

print(20*'-' + 'End Q5' + 20*'-')

# =================================================================
# Class_Ex6:
# Write a function that read two text files and combine each line from first
# text file with the corresponding line in second text file.
# Use T1.txt and T2.txt as sample text.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q6' + 20*'-')

def two_text(first_text, second_text):
    first = open(first_text, 'r', encoding='utf-8')
    second = open(second_text, 'r', encoding='utf-8')
    #with open(first_text, 'r', encoding='utf-8') as first, open(second_text, 'r', encoding='utf-8') as second:
    for line1, line2 in zip(first, second):
        return line1 + line2

print(two_text('T1.txt', 'T2.txt'))


print(20*'-' + 'End Q6' + 20*'-')
# =================================================================
# Class_Ex7:
# Write a function that create a text file where all letters of English alphabet
# put together by number of letters on each line (use n as argument in the function).
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q7' + 20*'-')

def all_letters(file_name, n):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    file = open(f'{file_name}.txt', 'w+')
    for i in range(n):
        file.write(letters[0:i] + '\n')

all_letters('new_file',7)


print(20*'-' + 'End Q7' + 20*'-')
# =================================================================
# Class_Ex8:
# Write a function that reads a text file and count number of words.
# Note: USe test_1.txt as a sample text.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q8' + 20*'-')

def count_words(file_name):
    file = open(f"{file_name}.txt", 'r', encoding='utf-8')
    string = ''
    for line in file:
        string += line
    return len(string.split())

print(count_words('test_1'))


print(20*'-' + 'End Q8' + 20*'-')
# =================================================================
# Class_Ex9:
# Write a script that go over over elements and repeat it each as many times as its count.
# Sample Output = ['o' ,'o', 'o', 'g' ,'g', 'f']
# Use Collections
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q9' + 20*'-')

from collections import Counter
c = Counter('sjidfnousnfsjkfn')
print(sorted(c.elements()))

print(20*'-' + 'End Q9' + 20*'-')
# =================================================================
# Class_Ex10:
# Write a program that appends couple of integers to a list
# and then with certain index start the the list over that index.
# Note: use deque
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q10' + 20*'-')

from collections import deque
from itertools import islice
list = ["a","b","c"]
deq = deque(list)
deq.append([1])
deq.append(7)
list2 = []
for i in range(deq.index('b'),deq.index(7)):
    list2.append(deq[i])
print(list2)


print(20*'-' + 'End Q10' + 20*'-')
# =================================================================
# Class_Ex11:
# Write a script using os command that finds only directories, files and all directories, files in a  path.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q11' + 20*'-')

import os
path = 'C:\\Users\\brear\\OneDrive\\Desktop\\NLP\\Lecture_01\\Class Ex'
dir_list = os.listdir(path)
print(dir_list)

print(20*'-' + 'End Q11' + 20*'-')
# =================================================================
# Class_Ex12:
# Write a script that create a file and write a specific text in it and rename the file name.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q12' + 20*'-')



file = open('fileQ12.txt', 'w')
file.write('This is the new message for question 12.')
file.close()
os.rename('fileQ12.txt', 'fileQ12_new.txt')

print(20*'-' + 'End Q12' + 20*'-')
# =================================================================
# Class_Ex13:
#  Write a script  that scan a specified directory find which is  file and which is a directory.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q13' + 20*'-')

path = 'C:\\Users\\brear\\OneDrive\\Desktop\\NLP\\Lecture_01\\Class Ex'
for instance in os.scandir(path):
    if instance.is_dir():
        typ = 'dir'
    elif instance.is_file():
        typ = 'file'
    print('{name} {typ}'.format(
        name=instance.name,
        typ=typ,
    ))


print(20*'-' + 'End Q13' + 20*'-')
# =================================================================
# ----------------------------------------------------------------
print(20*'-' + 'Begin E1' + 20*'-')
string = input()

print(sum(map(str.isupper, string)))
print(sum(map(str.islower, string)))
print(sum(map(str.isnumeric, string)))
#print(map(str.count(" "), input))







print(20*'-' + 'End E1' + 20*'-')

# =================================================================
# ----------------------------------------------------------------
print(20*'-' + 'Begin E2' + 20*'-')

string = '12345'
new_string = string[1:] + (string[0])
print(new_string)




print(20*'-' + 'End E2' + 20*'-')

# =================================================================
# ----------------------------------------------------------------
print(20*'-' + 'Begin E3' + 20*'-')

name = input()
list = name.split()
initials = []
for i in range(len(list)):
    initials.append(list[i][0])
print(initials)


print(20*'-' + 'End E3' + 20*'-')

# =================================================================
# ----------------------------------------------------------------
print(20*'-' + 'Begin E4' + 20*'-')

length = lower = upper = digit = False
password = input('Create new password: ')
if len(password) >= 8:
    length = True
    for letter in password:
        if letter.islower():
            lower = True
        elif letter.isupper():
            upper = True
        elif letter.isdigit():
            digit = True
if length and lower and upper and digit:
    print('That is a valid password.')
elif length or lower or upper or digit == False:
    print('That password is not valid.')
    password = input('Create new password: ')


print(20*'-' + 'End E4' + 20*'-')

# =================================================================
# ----------------------------------------------------------------
print(20*'-' + 'Begin E5' + 20*'-')

import operator
dict = {}
string = 'aaabbbbbbcccc'
for i in string:
    count = 0
    for j in string:
        if i == j:
            count += 1
    dict[i] = count
sorted_dict = sorted(dict.items(), key=operator.itemgetter(1))
print(sorted_dict)


print(20*'-' + 'End E5' + 20*'-')

# =================================================================
# ----------------------------------------------------------------
print(20*'-' + 'Begin E6' + 20*'-')

import itertools

string = 'abc'
print(map(''.join, itertools.product(*((x.lower(), x.upper()) for x in string))))



print(20*'-' + 'End E6' + 20*'-')

# =================================================================
# ----------------------------------------------------------------
print(20*'-' + 'Begin E7' + 20*'-')

from collections import Counter
text = open('test.txt' , 'w+', encoding='utf-8')
text.write('this is a test/n test file/n question/n random words')
print(next(text) for x in range(4))
print([str(text).split()].sort())
print(sum(1 for line in text))
print(Counter(str(text).split()))



print(20*'-' + 'End E7' + 20*'-')

# ==================================================================