import re
# ====================================Part 1================================
print(20*'-' + 'Part1' + 20*'-')
file = open('text.txt', 'r').read()


def f(text):
    pattern = re.compile(r'[a-z]+_[a-z]+')
    if re.search(pattern, text):
        return 'There is a match.'
    else:
        return ('no match.')


print(f("agb_cbjjc"))
print(f("agb_Abfjbbc"))
print(f("Adaf_abbbc"))
# ====================================Part 2================================
print(20*'-' + 'Part2' + 20*'-')

def f(text):
    pattern = re.compile(r'[A-Z].[a-z]+')
    if re.search(pattern, text):
        return 'There is a match.'
    else:
        return ('no match.')




print(f("aab_cbbbc"))
print(f("aab_Abbbc"))
print(f("Aaab_abbbc"))
# ====================================Part 3================================
print(20*'-' + 'Part3' + 20*'-')

def f(text):
    pattern = re.compile(r'\w+\S*$')
    if re.search(pattern, text):
        return 'There is a match.'
    else:
        return ('no match.')


print(f("The quick brown fox jumps over the lazy dog."))
print(f("The quick brown fox jumps over the lazy dog:"))
print(f("The quick brown fox jumps over the lazy dog!"))

# ====================================Part 4================================
print(20*'-' + 'Part4' + 20*'-')

def f(text):
    pattern = re.compile(r'\w*z.\w*')
    if re.search(pattern, text):
        return 'There is a match.'
    else:
        return ('no match.')



print(f("The quick brown fox jumps over the lazy dog."))
print(f("Regx Class."))

# ====================================Part 5================================
print(20*'-' + 'Part4' + 20*'-')
text = 'Python exercises, PHP exercises, C# exercises'
pattern = 'exercises'


pattern = 'exercises'
for match in re.findall(pattern, text):
    print(match)