print(20*'-' + 'Begin E1' + 20*'-')

# i
file =  open('Email.txt', 'r')
text = file.read()
file.close()
# ii
import re
pattern = re.compile(r'[a-zA-Z0-9._-]+@[a-zA-z]+\.(com|edu|net)')
matches = pattern.finditer(text)
for match in matches:
    print(match)
# iii
print(text)
print(20*'-' + 'End E1' + 20*'-')
print(20*'-' + 'Begin E2' + 20*'-')

# i
file = open('war_and_peace.txt', 'r')
text = file.read()
# ii
from collections import Counter
pattern = re.compile(r'[A-Z][a-zA-Z]+ski')
matches = pattern.finditer(text)
list = []
for match in matches:
    print(match)

# iii
list = []
for match in matches:
    list += [match[0]]
print(Counter(list))

print(20*'-' + 'End E2' + 20*'-')
print(20*'-' + 'Begin E3' + 20*'-')

# i

# ii
text = 'random string 234#$ (hello)'
print(text)
pattern = re.compile(r'\([a-zA-Z0-9]+\)')
matches = pattern.finditer(text)
for match in matches:
    length = len(match[0])-2
    text = re.sub(r'\([a-zA-Z0-9]+\)', '('+'x'*length+')', text)
print(text)
# iii
# iv
# v
# vi
# vii

print(20*'-' + 'End E3' + 20*'-')