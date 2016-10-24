
# coding: utf-8

# #### Saurabh Seth (sseth12@uw.edu)

# In[1]:

import numpy as np
import heapq
from scipy import spatial


# ##### RUNNING ANALOGY TEST 

# In[2]:

## Methods for calculating cosine similarity
def checkAnalogy(embedVectors, data):
    predictions=0
    for line in data:
        vect_d=embedVectors[line[1].lower()] - embedVectors[line[0].lower()] + embedVectors[line[2].lower()]
        max_cosine = -1.0
        max_key='';
        for w,vec_w in embedVectors.items():
            cos=1-spatial.distance.cosine(vec_w,vect_d)
            if max_cosine < cos:
                max_cosine=cos
                max_key=w
        if max_key==line[3].lower():
            predictions+=1
    return 100*(predictions/len(data))


# ###### Reduced Vocabulary

# In[3]:

red_vocab={}
for line in open('reduced_vocab.txt','r'):
    red_vocab[line.strip('\n')]=1
print('Length of Vocabulary -- ' + str(len(red_vocab)))


# In[4]:

## Read GloVe 50d Embeddings
red_embeddings={}
all_embeddings={}
for line in open('glove.6B.50d.txt', 'r'):
    tokens=line.strip('\n').split(' ')
    all_embeddings[tokens[0]] = np.array([float(i) for i in tokens[1:]]);
    if tokens[0] in red_vocab:
        red_embeddings[tokens[0]] = all_embeddings[tokens[0]]

print('Length of Total Embeddings for 50d -- ' + str(len(all_embeddings)))
print('Length of Reduced Embeddings for 50d -- ' + str(len(red_embeddings)))


# In[5]:

## Read GloVe 300d Embeddings
red_embeddings_300={}
all_embeddings_300={}
for line in open('glove.6B.300d.txt', 'r'):
    tokens=line.strip('\n').split(' ')
    all_embeddings_300[tokens[0]] = np.array([float(i) for i in tokens[1:]]);
    if tokens[0] in red_vocab:
        red_embeddings_300[tokens[0]] = all_embeddings_300[tokens[0]]

print('Length of Total Embeddings for 300d -- ' + str(len(all_embeddings_300)))
print('Length of Reduced Embeddings for 300d -- ' + str(len(red_embeddings_300)))


# ###### TEST ALL ANALOGIES WITH 2 FLAVORS - GloVe 50d & GloVe 300d

# In[6]:

## Read Capital Analogy 
analogy_capital=[]
for line in open('analogy/capital_world.txt','r'):
    lines=line.strip('\n').split(' ')
    analogy_capital.append([item.strip('\t') for item in lines])
print('Capital DataSet - ' + str(len(analogy_capital)) + ' Rows')
print('GloVe Embeddings 50D :: Capital DataSet - ' + str(checkAnalogy(red_embeddings,analogy_capital)) + '%')
print('GloVe Embeddings 300D :: Capital DataSet - ' + str(checkAnalogy(red_embeddings_300,analogy_capital)) + '%')


# In[8]:

## Read City Analogy 
analogy_city=[]
for line in open('analogy/city_in_state.txt','r'):
    lines=line.strip('\n').split(' ')
    analogy_city.append([item.strip('\t') for item in lines])
print('City DataSet - ' + str(len(analogy_city)) + ' Rows')
print('GloVe Embeddings 50D :: City DataSet - ' + str(checkAnalogy(red_embeddings,analogy_city)) + '%')
print('GloVe Embeddings 300D :: City DataSet - ' + str(checkAnalogy(red_embeddings_300,analogy_city)) + '%')


# In[9]:

## Read Currency Analogy 
analogy_currency=[]
for line in open('analogy/currency.txt','r'):
    lines=line.strip('\n').split(' ')
    analogy_currency.append([item.strip('\t') for item in lines])
print('Currency DataSet - ' + str(len(analogy_currency)) + ' Rows')
print('GloVe Embeddings 50D :: Currency DataSet - ' + str(checkAnalogy(red_embeddings,analogy_currency)) + '%')
print('GloVe Embeddings 300D :: Currency DataSet - ' + str(checkAnalogy(red_embeddings_300,analogy_currency)) + '%')


# In[10]:

## Read Family Analogy 
analogy_family=[]
for line in open('analogy/family.txt','r'):
    lines=line.strip('\n').split(' ')
    analogy_family.append([item.strip('\t') for item in lines])
print('Family DataSet - ' + str(len(analogy_family)) + ' Rows')
print('GloVe Embeddings 50D :: Family DataSet - ' + str(checkAnalogy(red_embeddings,analogy_family)) + '%')
print('GloVe Embeddings 300D :: Family DataSet - ' + str(checkAnalogy(red_embeddings_300,analogy_family)) + '%')


# In[11]:

## Read Adjective to Adverb Analogy 
analogy_adj_adv=[]
for line in open('analogy/gram1_adjective_to_adverb.txt','r'):
    lines=line.strip('\n').split(' ')
    analogy_adj_adv.append([item.strip('\t') for item in lines])
print('Adj to Adv DataSet - ' + str(len(analogy_adj_adv)) + ' Rows')
print('GloVe Embeddings 50D :: Adj to Adv DataSet - ' + str(checkAnalogy(red_embeddings,analogy_adj_adv)) + '%')
print('GloVe Embeddings 300D :: Adj to Adv DataSet - ' + str(checkAnalogy(red_embeddings_300,analogy_adj_adv)) + '%')


# In[12]:

## Read Opposite Analogy 
analogy_opp=[]
for line in open('analogy/gram2_opposite.txt','r'):
    lines=line.strip('\n').split(' ')
    analogy_opp.append([item.strip('\t') for item in lines])
print('Opposite DataSet - ' + str(len(analogy_opp)) + ' Rows')
print('GloVe Embeddings 50D :: Opposite DataSet - ' + str(checkAnalogy(red_embeddings,analogy_opp)) + '%')
print('GloVe Embeddings 300D :: Opposite DataSet - ' + str(checkAnalogy(red_embeddings_300,analogy_opp)) + '%')


# In[13]:

## Read Comparative Analogy 
analogy_com=[]
for line in open('analogy/gram3_comparative.txt','r'):
    lines=line.strip('\n').split(' ')
    analogy_com.append([item.strip('\t') for item in lines])
print('Comparative DataSet - ' + str(len(analogy_com)) + ' Rows')
print('GloVe Embeddings 50D :: Comparative DataSet - ' + str(checkAnalogy(red_embeddings,analogy_com)) + '%')
print('GloVe Embeddings 300D :: Comparative DataSet - ' + str(checkAnalogy(red_embeddings_300,analogy_com)) + '%')


# In[14]:

## Read National Adjective Analogy 
analogy_nat_adj=[]
for line in open('analogy/gram6_nationality_adjective.txt','r'):
    lines=line.strip('\n').split(' ')
    analogy_nat_adj.append([item.strip('\t') for item in lines])
print('National Adjective DataSet - ' + str(len(analogy_nat_adj)) + ' Rows')
print('GloVe Embeddings 50D :: Nationality Adj DataSet - ' + str(checkAnalogy(red_embeddings,analogy_nat_adj)) + '%')
print('GloVe Embeddings 300D :: Nationality Adj DataSet - ' + 
      str(checkAnalogy(red_embeddings_300,analogy_nat_adj)) + '%')


# ##### WORD SIMILARITY AND ANTONYMS

# In[17]:

antonym_test=['increase','enter','start','accept','maximum','receive','arrive']

for antonym in antonym_test:
    vect_d=all_embeddings[antonym]
    cos_pqueue = []
    for w,vec_w in all_embeddings.items():
        cos=1-spatial.distance.cosine(vec_w,vect_d)
        heapq.heappush(cos_pqueue,(cos,w))
    best_matches=heapq.nlargest(11,cos_pqueue)
    best_words=[word[1] for word in best_matches]
    print('Word -- ' + antonym + ', Best Matches -- ' + str(best_words[1:]))


# ##### ADVERSARIAL TEST DESIGN

# In[21]:

# Language Analogy
analogy_language=[['Spanish','Spain','German','Germany'],
                  ['Japan','Japanese','China','Mandarin'],
                 ['Spain','Spanish','Japan','Japanese']]

for line in analogy_language:
    vect_d=all_embeddings[line[1].lower()] - all_embeddings[line[0].lower()] + all_embeddings[line[2].lower()]
    max_cosine = -1.0
    max_key='';
    for w,vec_w in all_embeddings.items():
        cos=1-spatial.distance.cosine(vec_w,vect_d)
        if max_cosine < cos:
            max_cosine=cos
            max_key=w
    print('Line --' + str(line))
    print('Expected -- ' + str(line[3].lower()) + ' :: Received -- ' + str(max_key))
    print()


# In[22]:

# Verb Object Analogy
analogy_verbObj=[['Swimming','Water','Flying','Air'],
                  ['Walking','Floor','Swimming','Pool']]

for line in analogy_verbObj:
    vect_d=all_embeddings[line[1].lower()] - all_embeddings[line[0].lower()] + all_embeddings[line[2].lower()]
    max_cosine = -1.0
    max_key='';
    for w,vec_w in all_embeddings.items():
        cos=1-spatial.distance.cosine(vec_w,vect_d)
        if max_cosine < cos:
            max_cosine=cos
            max_key=w
    print('Line --' + str(line))
    print('Expected -- ' + str(line[3].lower()) + ' :: Received -- ' + str(max_key))
    print()


# In[23]:

# Singular Plural Analogy
analogy_plural=[['Man','Men','Woman','Women'],
                ['Ball','Balls','Fruit','Fruits'],
                ['Bird','Birds','Animal','Animals']]

for line in analogy_plural:
    vect_d=all_embeddings[line[1].lower()] - all_embeddings[line[0].lower()] + all_embeddings[line[2].lower()]
    max_cosine = -1.0
    max_key='';
    for w,vec_w in all_embeddings.items():
        cos=1-spatial.distance.cosine(vec_w,vect_d)
        if max_cosine < cos:
            max_cosine=cos
            max_key=w
    print('Line --' + str(line))
    print('Expected -- ' + str(line[3].lower()) + ' :: Received -- ' + str(max_key))
    print()


# In[24]:

# Superlative Forms Analogy
analogy_superlative=[['Tall','Taller','Short','Shorter'],
                ['Tight','Tighter','Bright','Brighter'],
                ['Large','Largest','Big','Biggest']]

for line in analogy_superlative:
    vect_d=all_embeddings[line[1].lower()] - all_embeddings[line[0].lower()] + all_embeddings[line[2].lower()]
    max_cosine = -1.0
    max_key='';
    for w,vec_w in all_embeddings.items():
        cos=1-spatial.distance.cosine(vec_w,vect_d)
        if max_cosine < cos:
            max_cosine=cos
            max_key=w
    print('Line --' + str(line))
    print('Expected -- ' + str(line[3].lower()) + ' :: Received -- ' + str(max_key))
    print()

