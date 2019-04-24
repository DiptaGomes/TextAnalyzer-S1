import nltk
import pandas as pd
import numpy as np
import csv
from nltk.tokenize import word_tokenize

def bigram_generation(tokens):
    print("Bigrams:")
    bigrm = nltk.bigrams(tokens)
    print(*map(' '.join, bigrm), sep=' , ')
    print(tokens)

def pos_tagging(tokens):
    print("After POS tagging")
    pos_tag = nltk.pos_tag(tokens)
    print(pos_tag)

def lemmatization(string):
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    print("After Lemmatization")
    nltk_tokens = nltk.word_tokenize(string)
    for w in nltk_tokens:
        print("Actual: %s  Lemma: %s" % (w, wordnet_lemmatizer.lemmatize(w)))

def Stemming(string):
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    print("After stemming")
    nltk_tokens = nltk.word_tokenize(string)
    # Next find the roots of the word
    for w in nltk_tokens:
        print("Actual: %s  Stem: %s" % (w, porter.stem(w)))

# Read from source
com = pd.read_csv("comments.csv")

#convert dataframe to string
com1 = com.loc[com['serial'] == 9]
com2 = com1.corpus.astype(str)
main_string = str(com2)
com2=str(com2)

# Token GEneration
tokens = nltk.word_tokenize(com2)

#Bigram Generation
bigram_generation(tokens)

# POS tagging
pos_tagging(tokens)

# lemmatization
lemmatization(com2)

#stemming
Stemming(com2)