
# coding: utf-8

# ## HOMEWORK 5 - Structured Learning with Perceptrons

# #### Saurabh Seth (sseth12@uw.edu)

# In[1]:

from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
import math
import nltk
import glob
import json
import re
import numpy as np
import time
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool


# ##### Method to generate features from the data

# In[2]:

def feature_generation(data):
    feature_number = 0;
    features = {};
    nertag_dict = {};
    total_feature_cnt = defaultdict(int);
    
    for line in data:
        if len(line) == 0:
            continue;
        sentence = [];
        nertags = [];
        postags = [];
        chunktags = [];
        for words in line:
            tokens = words.split(' ');
             
            try:
                float(tokens[0])
                word = '#float';
            except ValueError:
                word = tokens[0]
            
            sentence.append(word);
            postags.append(tokens[1]);
            chunktags.append(tokens[2]);
            nertags.append(tokens[3][:-1]);
            nertag_dict[tokens[3][:-1]] = 1;
        
        ## Feature: w_-1 and w_+1
        ## Feature: current word
        prevWord = '<start>';
        word = sentence[0];
        ## Feature: POS Tag
        prevPOSTag = '<start>';
        currentPOS = postags[0];
        ## Feature: NER Tag
        prevNERTag = '<start>';
        for index,tag in enumerate(nertags):
            if index == len(nertags) -1 :
                nextWord = '<stop>';
                nextPOSTag = '<stop>';
                if ('<stop>','#Bigram',tag) not in features:
                    feature_number += 1;
                    features[('<stop>','#Bigram',tag)] = feature_number;
            else:
                nextWord = sentence[index+1];
                nextPOSTag = postags[index+1];
            
            ## Feature: w_-1 and w_0
            if (tag,'#WordPair',(prevWord,word)) not in features:
                feature_number += 1
                features[(tag,'#WordPair',(prevWord,word))] = feature_number;
                total_feature_cnt[(tag,'#WordPair',(prevWord,word))] += 1;
            
            ## Feature: w_0
            if (tag,'#Word',word) not in features:
               feature_number += 1;
               features[(tag,'#Word',word)] = feature_number;
               total_feature_cnt[(tag,'#Word',word)] += 1;
            
            ## Feature: p_-1 and p_+1
            ##if (tag,'#POSPair',(prevPOSTag,nextPOSTag)) not in features:
            ##    feature_number += 1;
            ##    features[(tag,'#POSPair',(prevPOSTag,nextPOSTag))] = feature_number;
            ##    total_feature_cnt[(tag,'#POSPair',(prevPOSTag,nextPOSTag))] += 1;
                
            ## Feature: p_0
            if (tag,'#POSTag',currentPOS) not in features:
                feature_number += 1;
                features[(tag,'#POSTag',postags[index])] = feature_number;
                total_feature_cnt[(tag,'#POSTag',postags[index])] += 1;
            
            ## Feature: ChunkTag
            if (tag,'#Chunk',chunktags[index]) not in features:
                feature_number += 1;
                features[(tag,'#Chunk',chunktags[index])] = feature_number;
                total_feature_cnt[(tag,'#Chunk',chunktags[index])] += 1;
            
            ## Feature: NERTag
            if (tag,'#Bigram',prevNERTag) not in features:
                feature_number += 1;
                features[(tag,'#Bigram',prevNERTag)] = feature_number;
            
            prevWord = word;
            word = nextWord;
            prevPOSTag = currentPOS;
            currentPOS = nextPOSTag;
            prevNERTag = tag;
    
    ## First word is capital
    #for nertag in nertag_dict:
    #    if (nertag,'#FirstCapital',True) not in features:
    #        feature_number += 1;
    #        features[(nertag,'#FirstCapital',True)] = feature_number;
    
    #for key,value in total_feature_cnt.items():
    #    if value < 3:
    #        features.pop(key, None)
    
    return features, nertag_dict;


# ##### Read training data

# In[3]:

with open("conll03_ner/eng.train", "r") as data:
    trainDataArray = []
    i = 0;
    trainDataArray.append([])
    for line in data:
        if line == '\n':
            i = i + 1;
            trainDataArray.append([])
        else:
            trainDataArray[i].append(line);
print(len(trainDataArray))


# ##### Method call to get all the features from the sentence

# In[4]:

def feature_vector(pastNER,currentNER,sentence,index,features,weight):
    prev_word ='<start>';
    next_word = '<stop>';
    prev_tag = '<start>';
    next_tag = '<stop>';
    word = '';
    total_weight = 0.0;
    feature_index=[];
    
    if index > 0:
        tokens = sentence[index-1].split(' ');
        try:
            float(tokens[0])
            prev_word = '#float';
        except ValueError:
            prev_word = tokens[0];
        prev_tag = tokens[1];
    
    if index != len(sentence)-1:
        tokens = sentence[index + 1].split(' ');
        try:
            float(tokens[0])
            next_word = '#float';
        except ValueError:
            next_word = tokens[0]
        next_tag = tokens[1];
        
    
    tokens = sentence[index].split(' ');
    try:
        float(tokens[0])
        word = '#float';
    except ValueError:
        word = tokens[0]
    currentPOS = tokens[1];
    chunktag = tokens[2];
    
    if (currentNER,'#WordPair',(prev_word,word)) in features:
        feature = features[(currentNER,'#WordPair',(prev_word,word))]
        feature_index.append(feature)
        total_weight += weight[feature];
    if (currentNER,'#Word',word) in features:
        feature = features[(currentNER,'#Word',word)];
        feature_index.append(feature)
        total_weight += weight[feature];
    #if (currentNER,'#POSPair',(prev_tag,next_tag)) in features:
    #    feature = features[(currentNER,'#POSPair',(prev_tag,next_tag))];
    #    feature_index.append(feature)
    #    total_weight += weight[feature];
    if (currentNER,'#POSTag',currentPOS) in features:
        feature = features[(currentNER,'#POSTag',currentPOS)];
        feature_index.append(feature)
        total_weight += weight[feature];
    if (currentNER,'#Chunk',chunktag) in features:
        feature = features[(currentNER,'#Chunk',chunktag)];
        feature_index.append(feature)
        total_weight += weight[feature];
    #if word[0].isupper():
    #    if (currentNER,'#FirstCapital',True) in features:
    #        feature = features[(currentNER,'#FirstCapital',True)];
    #        feature_index.append(feature)
    #        total_weight += weight[feature];
    if (currentNER,'#Bigram',pastNER) in features:
        feature = features[(currentNER,'#Bigram',pastNER)];
        feature_index.append(feature)
        total_weight += weight[feature];
    
    return total_weight,feature_index;


# ##### VITERBI HMM FOR RETRIEVING BEST TAG

# In[5]:

def viterbi_hmm(tags,line,features,weight):
    columns = len(line);
    rows = len(tags);
    
    probs = np.zeros((rows, columns));
    paths = np.zeros((rows,columns),np.int8);
    paths[:, 0] = -1;
    best_prob = float("-inf");
    best_index = 0;
    
    for row,tag in enumerate(tags):
        probs[row][0],indexes = feature_vector('<start>',tag,line,0,features,weight);
    
    for index in range(1,columns):
        for row,tag in enumerate(tags):
            max_prob = float("-inf");
            tagIndex = 0
            for internalRow,oldTag in enumerate(tags):
                temp_prob,indexes = feature_vector(oldTag,tag,line,index,features,weight);
                temp_prob += probs[internalRow][index-1];
                if max_prob < temp_prob:
                    max_prob = temp_prob;
                    tagIndex = internalRow;
            probs[row][index] = max_prob;
            paths[row][index] = tagIndex;
    
    for row,tag in enumerate(tags):
        if ('<stop>','#Bigram',tag) in features:
            temp_prob = weight[features[('<stop>','#Bigram',tag)]];
            temp_prob += probs[row][columns-1];
            if best_prob < temp_prob:
                best_prob = temp_prob;
                best_index = row;
    
    best_tag = [];
    for column in reversed(range(0,columns)):
        best_tag.insert(0,tags[best_index]);
        best_index=paths[best_index][column];
        
    return best_tag;


# ##### Method call to generate features and tags

# In[6]:

features,tagdict = feature_generation(trainDataArray);
tags=tagdict.keys();
tags=sorted(tags);
weight = defaultdict(int);
#count = 0;
print(tags)
print(len(features))
#print(features[()])
#index = 0;
#for key in features:
#    index = index + 1;
#    print(key,features[key])
#    if index > 10:
#        break;


# In[7]:

def getWeights(line):
    if len(line) == 0:
        return;
    orig_weight_indices = [];
    pred_weight_indices = [];
    origTag = [words.split(' ')[3][:-1] for words in line];
    predTag = viterbi_hmm(tags,line,features,weight);
    count = 0;
    
    #print(','.join(origTag))
    #print(','.join(predTag))
    
    if ','.join(origTag) != ','.join(predTag):
        count += 1;
        prevTag = '<start>';
        for row,tag in enumerate(origTag):
            total_weight, weight_index = feature_vector(prevTag,tag,line,row,features,weight);
            orig_weight_indices.extend(weight_index);
            prevTag = tag;
        prevPredTag = '<start>';
        for row,tag in enumerate(predTag):
            total_weight, weight_index = feature_vector(prevPredTag,tag,line,row,features,weight);
            pred_weight_indices.extend(weight_index);
            prevPredTag = tag;
            
    return orig_weight_indices,pred_weight_indices,count;


# #### DEVELOPMENT DATA

# In[8]:

with open("conll03_ner/eng.dev", "r") as data:
    devDataArray = []
    i = 0;
    devDataArray.append([])
    for line in data:
        if line == '\n':
            i = i + 1;
            devDataArray.append([])
        else:
            devDataArray[i].append(line);
print(len(devDataArray))


# ### TRAINING WEIGHTS

# In[9]:

weight = defaultdict(int);
total_weight = defaultdict(int);
T = 2500;
total_count = 0;
start_time = time.time();
for t in range(1,T):
    p = Pool(50);
    weight_indices=[]
    weight_indices += p.map(getWeights, trainDataArray)
    p.close()
    p.join()
    
    #print(len(weight_indices))
    for weights in weight_indices:
        if weights is not None:
            (orig_weight,pred_weight,count) = weights;
            for index in pred_weight:
                weight[index] -= 1;
            #print(len(orig_weight))
            for index in orig_weight:
                weight[index] += 1;
            total_count += count;
    
    for key in weight:
        total_weight[key] += weight[key]

    print(time.time()-start_time)
    
    if t % 5 == 0:
        testWeight = defaultdict(float);
        for x,y in total_weight.items():
            testWeight[x] = y/total_count;
        
        filename = 'dev/dev_test' + str(t) + '.txt';
        file = open(filename, 'w');
        for dev in devDataArray:
            if len(dev) == 0:
                file.write("\n");
                continue;
            preddevTag = viterbi_hmm(tags,dev,features,testWeight);
            
            index = 0;
            for words in dev:
                line = words[:-1] + ' ' + preddevTag[index] + '\n';
                index = index+1;
                file.write(line)
                file.write("\n")
            
        file.close()
        print(t,'Iterations Complete. Time Time -- ' + str(time.time()-start_time))
        print('Output written to file on Development Set -- ',filename);

avgWeight = defaultdict(float);
for x,y in total_weight.items():
    avgWeight[x] = y/total_count;
#print(weight)


# #### Testing on Dev Set

# In[10]:

filename = 'devFinal.txt';
file = open(filename, 'w');
for dev in devDataArray:
    if len(dev) == 0:
        file.write("\n");
        continue;
    preddevTag = viterbi_hmm(tags,dev,features,avgWeight);
            
    index = 0;
    for words in dev:
        line = words[:-1] + ' ' + preddevTag[index] + '\n';
        index = index+1;
        file.write(line)
        file.write("\n")   
file.close()


# #### Testing on Test Set

# In[11]:

with open("conll03_ner/eng.test", "r") as data:
    testDataArray = []
    i = 0;
    testDataArray.append([])
    for line in data:
        if line == '\n':
            i = i + 1;
            testDataArray.append([])
        else:
            testDataArray[i].append(line);
print(len(testDataArray))


# In[12]:

filename = 'testFinal.txt';
file = open(filename, 'w');
for dev in testDataArray:
    if len(dev) == 0:
        file.write("\n");
        continue;
    preddevTag = viterbi_hmm(tags,dev,features,avgWeight);
            
    index = 0;
    for words in dev:
        line = words[:-1] + ' ' + preddevTag[index] + '\n';
        index = index+1;
        file.write(line)
        file.write("\n")
file.close()

