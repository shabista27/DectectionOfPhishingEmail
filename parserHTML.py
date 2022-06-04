from bs4 import BeautifulSoup
import os
direc ="2016/"
files=os.listdir(direc)

emails=[direc+email for email in files]

words=[]
c = len(emails)
for email in emails:
    data=[]

    f = open(email)
    
    blob = f.read()
    words += blob.split(" ")
    print (c)
    soup = BeautifulSoup(blob, features="lxml").text
    print(soup)
    #print(blob)