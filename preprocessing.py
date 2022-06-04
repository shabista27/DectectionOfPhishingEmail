import os
from collections import Counter
from email.parser import Parser
parser = Parser()


    
direc = "2016-/"
files = os.listdir(direc)
emails = [direc + email for email in files]
words = []
c = len(emails)
for email in emails:
    f = open(email)
    blob = f.read()
    words += blob.split(" ")
    print(c)
    c -= 1

for i in range(len(words)):
    if not words[i].isalpha():
        words[i] = ""

dictionary = Counter(words)
del dictionary[""]
print(dictionary.most_common(300))

#print (words)
