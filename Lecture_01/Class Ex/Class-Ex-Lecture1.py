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
    return list(first.readline()).append(second.readline())

print(two_text('T1.txt', 'T2.txt'))







print(20*'-' + 'End Q6' + 20*'-')
# =================================================================
# Class_Ex7:
# Write a function that create a text file where all letters of English alphabet
# put together by number of letters on each line (use n as argument in the function).
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q7' + 20*'-')








print(20*'-' + 'End Q7' + 20*'-')
# =================================================================
# Class_Ex8:
# Write a function that reads a text file and count number of words.
# Note: USe test_1.txt as a sample text.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q8' + 20*'-')







print(20*'-' + 'End Q8' + 20*'-')
# =================================================================
# Class_Ex9:
# Write a script that go over over elements and repeat it each as many times as its count.
# Sample Output = ['o' ,'o', 'o', 'g' ,'g', 'f']
# Use Collections
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q9' + 20*'-')

import itertools
states = ['Newyork', 'Virginia', 'DC', 'Texas']

#result = itertools.coll






print(20*'-' + 'End Q9' + 20*'-')
# =================================================================
# Class_Ex10:
# Write a program that appends couple of integers to a list
# and then with certain index start the the list over that index.
# Note: use deque
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q10' + 20*'-')









print(20*'-' + 'End Q10' + 20*'-')
# =================================================================
# Class_Ex11:
# Write a script using os command that finds only directories, files and all directories, files in a  path.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q11' + 20*'-')







print(20*'-' + 'End Q11' + 20*'-')
# =================================================================
# Class_Ex12:
# Write a script that create a file and write a specific text in it and rename the file name.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q12' + 20*'-')









print(20*'-' + 'End Q12' + 20*'-')
# =================================================================
# Class_Ex13:
#  Write a script  that scan a specified directory find which is  file and which is a directory.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q13' + 20*'-')









print(20*'-' + 'End Q13' + 20*'-')
# =================================================================
# Class_Ex14:
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q14' + 20*'-')








print(20*'-' + 'End Q14' + 20*'-')

# =================================================================
# Class_Ex15:
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q15' + 20*'-')








print(20*'-' + 'End Q15' + 20*'-')