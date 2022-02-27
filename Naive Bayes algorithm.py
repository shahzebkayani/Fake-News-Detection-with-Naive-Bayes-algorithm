
from __future__ import division
from os import listdir

import re, numpy as np, operator
from nltk.corpus import stopwords

from functools import reduce

STOPWORDS = stopwords.words('english')
real1 = [open('data/Real/'+i, "r").read() for i in listdir('data/Real')] #  real data
fake1 = [open('data/Fake/'+i, "r").read() for i in listdir('data/Fake')] # fake data

REAL = real1[:int(.7*len(real1))]
FAKE = fake1[:int(.7*len(fake1))]
test_real = real1[-1*(len(real1)-int(.7*len(real1))):]
test_fake = fake1[-1*(len(fake1)-int(.7*len(fake1))):]
ALL = REAL+FAKE

# Tokenization
def words(text): 
    return re.findall(r'\w+', text.lower())
def tokenize(text): 
    return [word for word in words(text) if word not in STOPWORDS]

real_words = []
fake_words = []
all_words = []

for i in [tokenize(j) for j in REAL]:
        all_words += i
        real_words += i
for i in [tokenize(j) for j in FAKE]:
        all_words += i
        fake_words += i
print(all_words)
P_REAL = len(REAL)/len(ALL)
P_FAKE = 1 - P_REAL

def P(word, cat): return cat.count(word) / len(cat)

def fake(word):
        if (word not in fake_words):
                return 1
        P_w = P(word, fake_words)
        NP_w = P(word, real_words)
        return (P_w*P_FAKE)/(P_w*P_FAKE + NP_w*P_REAL+1e-10)

def real(word):
        if (word not in real_words):
                return 1
        P_w = P(word, real_words)
        NP_w = P(word, fake_words)
        return (P_w*P_REAL)/(P_w*P_REAL + NP_w*P_FAKE+1e-10)



def product(a):
        return reduce(operator.mul, a)

def predict(doc):

        fake_likelihood = product([fake(w) for w in tokenize(doc)])
        real_likelihood = product([real(w) for w in tokenize(doc)])
        #print(real_likelihood)
        prediction = np.argmax(np.asarray([fake_likelihood, real_likelihood]))
        return prediction

def test():
        count = 0
        false_p = 0 # Number reals classified as positive fakes
        false_n = 0 # Number of fakes negatively classified as reals
        for i in range(len(test_real)):
                if predict(test_real[i]) == 1:
                        count += 1
                else:
                        false_p += 1
                        
        for i in range(len(test_fake)):
                if predict(test_fake[i]) == 0:
                        count += 1
                else:
                        false_n += 1

        #print(fake_p)
        print ("False Negative Rate: " + str(false_n / len(test_fake)*100))
        
        print ("False Positive Rate: " + str(false_p / len(test_real)))
        print ("Accuracy: " + str(count / len(test_fake+test_real)))
test()