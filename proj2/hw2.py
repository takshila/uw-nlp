
# coding: utf-8

# ## HOMEWORK 2

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
from sklearn.metrics import confusion_matrix


# ###### METHODS FOR CLEANING DATA IN A SENTENCE

# In[2]:

def clean(sentence):
    cleanPairs = []
    for pairs in sentence:
        cleanPair = replaceUnkEmoticons(pairs[0]);
        cleanPair = replaceUrlLinks(cleanPair);
        cleanPair = replaceUserNames(cleanPair);
        cleanPair = replaceHashTags(cleanPair);
        cleanPair = replaceNumbers(cleanPair);
        cleanPairs.append([cleanPair,pairs[1]]);
    return cleanPairs;
    
def replaceUnkEmoticons(word):
    if re.findall('[\w+][\U00002600-\U0001f6ff]+',word):
        return re.sub('[\U00002600-\U0001f6ff]+','',word);
    elif re.findall('[\U00002600-\U0001f6ff]+',word):
        return '</emoticons>';
    return word;

def replaceUrlLinks(word):
    return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','</url>',word);

def replaceUserNames(word):
    return re.sub('@\w+','</username>',word);

def replaceHashTags(word):
    return re.sub('#\w+','</hashtag>',word);

def replaceNumbers(word):
    if re.findall(r'[a-zA-Z]+[0-9]+',word):
        return word;
    elif re.findall(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?',word):
        return '</number>';
    return word;


# ###### METHODS FOR GENERATING TAGS IN TRAINING DATA

# In[3]:

def generateTags(sentence,word_counts,tag_map,tags):
    line = ' ';
    for pairs in sentence:
        line += pairs[1];
        line += ' ';
        
        countTag = 0;
        if pairs[1] in tag_map:
            countTag = tag_map[pairs[1]];
        else :
            tags.append(pairs[1])
        tag_map[pairs[1]] = countTag + 1;
        
        # map of map for every word in the sentence.
        internal = {};
        tag_count = 0;
        
        if pairs[0] in word_counts:
            internal= word_counts[pairs[0]];
            if pairs[1] in internal:
                tag_count = internal[pairs[1]];
        
        internal[pairs[1]] = tag_count + 1;
        word_counts[pairs[0]] = internal;
    return line.strip(),word_counts,tag_map,tags;


# ###### METHODS FOR CALCULATING BIGRAM TAGS IN THE DATA

# In[4]:

def bigram_count(data):
    bigram_dict = {}
    unigram_dict = defaultdict(int);
    
    for line in data:
        sentence = '<start> ' + line + ' </stop>';
        words = sentence.split(' ');
        bigrams = tuple(nltk.bigrams(words));
        
        for word in words:
            unigram_dict[word] += 1;
        for bigram in bigrams:
            word1,word2 = bigram;
            if word2 in bigram_dict:
                internalDict = bigram_dict[word2];
                internalDict[word1] += 1
                bigram_dict[word2] = internalDict
            else:
                internalDict = defaultdict(int);
                internalDict[word1] += 1
                bigram_dict[word2] = internalDict 
    return bigram_dict,unigram_dict;

def bigram_prob(bigram_map, unigram_map):
    probs = {}
    probs = defaultdict(int);
    for word2 in bigram_map:
        probs[word2] = defaultdict(int);
        for word1 in bigram_map[word2]:
            probs[word2][word1]=bigram_map[word2][word1]/unigram_map[word1];
    return probs;


# ###### METHODS FOR CALCULATING TRIGRAM TAGS IN THE DATA

# In[5]:

def trigram_count(data):
    trigram_dict = defaultdict(int);
    bigram_dict = defaultdict(int);
    
    for line in data:
        sentence = '<start> <start> ' + line + ' </stop>';
        words = sentence.split(' ');
        bigrams = tuple(nltk.bigrams(words));
        trigrams = tuple(nltk.trigrams(words));
        
        for trigram in trigrams:
            trigram_dict[trigram] += 1;
        for bigram in bigrams:
            bigram_dict[bigram] += 1;
            
    return trigram_dict,bigram_dict;

def trigram_prob(trigram_map, bigram_map):
    probs = defaultdict(int);
    for word in trigram_map:
        word1,word2,word3 = word;
        probs[word]=trigram_map[word]/bigram_map[(word1,word2)];
    return probs;


# ###### EMISSION PROBABILITIES

# In[6]:

def emission_prob(data,tags):
    probs = {};
    for word in data:
        data_tags = data[word];
        tag_probs = defaultdict(int);
        for data_tag in data_tags:
            tag_probs[data_tag] = data_tags[data_tag]/tags[data_tag];
        probs[word]=tag_probs
    return probs;


# ### VITERBI DECODING FOR BIGRAM HMM

# In[7]:

def bigram_viterbi(emissions,transitions,tags,line):
    
    columns = len(line);
    rows = len(tags);
    
    probs = np.zeros((rows, columns));
    paths = np.zeros((rows,columns),np.int8);
    paths[:, 0] = -1;
    best_prob = 0.0;
    best_index = 0;
    
    for row,tag in enumerate(tags):
        probs[row][0] = transitions[tag]['<start>']*emissions[line[0]][tag];
    
    for index in range(1,columns):
        for row,tag in enumerate(tags):
            max_prob = 0.0
            tagIndex = 0
            emission_prob = emissions[line[index]][tag];
            for internalRow,oldTag in enumerate(tags):
                temp_prob = transitions[tag][oldTag]*probs[internalRow][index-1]*emission_prob;
                if max_prob < temp_prob:
                    max_prob = temp_prob;
                    tagIndex = internalRow;
            probs[row][index] = max_prob;
            paths[row][index] = tagIndex;
    
    for row,tag in enumerate(tags):
        temp_prob = transitions['</stop>'][tag]*probs[row][columns-1];
        if best_prob < temp_prob:
            best_prob = temp_prob;
            best_index = row;
    
    best_tag = [];
    for column in reversed(range(0,columns)):
        best_tag.insert(0,tags[best_index]);
        best_index=paths[best_index][column];
        
    return best_tag;


# In this algorithm, we are using a matrix of size (n * K) where n is the length of sequence and K are the number of tags. We are storing best score to reach to current tag considering all previous tags. 
# The running time for the algorithm is --
# 
# $$ O(n|K|^2)$$
# 
# Hence it is linear in the length of the sequence, and quadratic in the number of tags.

# ### VITERBI DECODING FOR TRIGRAM HMM

# In[8]:

def trigram_viterbi(emissions,transitions,tags,line):
    
    bigram_tags = [];
    for tag in tags:
        for oldTag in tags:
            bigram_tags.append((oldTag,tag));
    
    columns = len(line);
    count_tag = len(tags);
    rows = len(bigram_tags);
    
    probs = np.zeros((rows, columns));
    paths = np.zeros((rows,columns),np.int16);
    paths[:, 0] = -1;
    best_prob = 0.0;
    best_index = 0;
    
    for row,bigram_tag in enumerate(bigram_tags):
        word1,predTag = bigram_tag;
        trigram_start = ('<start>','<start>',predTag);
        probs[row][0] = transitions[trigram_start] * emissions[line[0]][predTag];
    
    indexer = 0;
    for row,bigram_tag in enumerate(bigram_tags):
        word1,predTag = bigram_tag;
        trigram_second = ('<start>',word1,predTag);
        prev_index = indexer*count_tag;
        probs[row][1] = transitions[trigram_second]*emissions[line[1]][predTag]*probs[prev_index][0];
        paths[row][1] = prev_index;
        indexer = indexer + 1;
        if indexer >= count_tag:
            indexer = 0;
    
    for index in range(2,columns):
        total_index = 0;
        for row,bigram_tag in enumerate(bigram_tags):
            word1,predTag = bigram_tag;
            max_prob = 0.0;
            internalIndex = 0;
            for internalRow,oldTag in enumerate(tags):
                trigram_tag = (oldTag,word1,predTag);
                temp_prob=transitions[trigram_tag]*emissions[line[index]][predTag]*probs[total_index][index-1];
                if max_prob < temp_prob:
                    max_prob = temp_prob;
                    internalIndex = total_index;
                total_index += 1;
            probs[row][index] = max_prob;
            paths[row][index] = internalIndex;
            if total_index >= len(bigram_tags):
                total_index = 0;
        
    
    for row,bigram_tag in enumerate(bigram_tags):
        word1,predTag = bigram_tag;
        trigram_end = (word1,predTag,'</stop>');
        temp_prob = transitions[trigram_end]*probs[row][columns-1];
        if best_prob < temp_prob:
            best_prob = temp_prob;
            best_index = row;
    
    # VITERBI DECODING
    best_tag = [];
    for column in reversed(range(0,columns)):
        word1,word2=bigram_tags[best_index];
        best_tag.insert(0,word2);
        best_index=paths[best_index][column];
    
    return best_tag;


# In this algorithm, we are using a matrix of size (n * K^2) where n is the length of sequence and K are the number of tags. We are storing best score to reach to current tag considering all previous tags. But we still need to consider only K transitions into each cell, since the current word's tag is the next word's preceding tag.
# The running time for the algorithm is --
# 
# $$ O(n|K|^3)$$
# 
# Hence it is linear in the length of the sequence, and cubic in the number of tags.

# ### VITERBI DECODING FOR LINEAR INTERPOLATION TRIGRAM HMM

# In[9]:

def trigram_viterbi_inter(emissions,transitionModel,tags,line):
    
    bigram_tags = [];
    for tag in tags:
        for oldTag in tags:
            bigram_tags.append((oldTag,tag));
    
    columns = len(line);
    count_tag = len(tags);
    rows = len(bigram_tags);
    
    probs = np.zeros((rows, columns));
    paths = np.zeros((rows,columns),np.int16);
    paths[:, 0] = -1;
    best_prob = 0.0;
    best_index = 0;
    
    for row,bigram_tag in enumerate(bigram_tags):
        word1,predTag = bigram_tag;
        trigram_start = ('<start>','<start>',predTag);
        probs[row][0] = transitionModel[trigram_start] * emissions[line[0]][predTag];
    
    indexer = 0;
    for row,bigram_tag in enumerate(bigram_tags):
        word1,predTag = bigram_tag;
        trigram_second = ('<start>',word1,predTag);
        prev_index = indexer*count_tag;
        probs[row][1] = transitionModel[trigram_second]*emissions[line[1]][predTag]*probs[prev_index][0];
        paths[row][1] = prev_index;
        indexer = indexer + 1;
        if indexer >= count_tag:
            indexer = 0;
    
    for index in range(2,columns):
        total_index = 0;
        for row,bigram_tag in enumerate(bigram_tags):
            word1,predTag = bigram_tag;
            max_prob = 0.0;
            internalIndex = 0;
            for internalRow,oldTag in enumerate(tags):
                trigram_tag = (oldTag,word1,predTag);
                temp_prob=transitionModel[trigram_tag]*emissions[line[index]][predTag]*probs[total_index][index-1];
                if max_prob < temp_prob:
                    max_prob = temp_prob;
                    internalIndex = total_index;
                total_index += 1;
            probs[row][index] = max_prob;
            paths[row][index] = internalIndex;
            if total_index >= len(bigram_tags):
                total_index = 0;
        
    
    for row,bigram_tag in enumerate(bigram_tags):
        word1,predTag = bigram_tag;
        trigram_end = (word1,predTag,'</stop>');
        temp_prob = transitionModel[trigram_end]*probs[row][columns-1];
        if best_prob < temp_prob:
            best_prob = temp_prob;
            best_index = row;
    
    # VITERBI DECODING
    best_tag = [];
    for column in reversed(range(0,columns)):
        word1,word2=bigram_tags[best_index];
        best_tag.insert(0,word2);
        best_index=paths[best_index][column];
    
    return best_tag;


# In this algorithm, we are using a matrix of size (n * K^2) where n is the length of sequence and K are the number of tags. We are storing best score to reach to current tag considering all previous tags. But we still need to consider only K transitions into each cell, since the current word's tag is the next word's preceding tag.
# The running time for the algorithm is --
# 
# $$ O(n|K|^3)$$
# 
# Hence it is linear in the length of the sequence, and cubic in the number of tags.

# #### TESTING METHODS

# In[10]:

def getTestSentence(test,vocabulary,count):
    originalTag = [];
    line = [];
    
    for pairs in test:
        if pairs[0] in vocabulary:
            line.append(pairs[0]);
        else:
            line.append('</unk>');
            count += 1
        originalTag.append(pairs[1]);
    
    return originalTag,line,count;


# In[11]:

def checkTags(bestTags,originalTags):
    total_count = 0;
    correct_count = 0;
    for extIndex,bestTag in enumerate(bestTags):
        total_count += len(bestTag);
        correct_count += len(bestTag);
        for index,tag in enumerate(bestTag):
            if tag != originalTags[extIndex][index]:
                correct_count -= 1;
    return correct_count,total_count;


# #### TRAINING

# In[12]:

def generateVocab(line,vocabulary):
    pairs = []
    for pair in line:
        if pair[0] not in vocabulary:
            vocabulary[pair[0]] = 1;
            pair[0] = '</unk>';
        else:
            vocabulary[pair[0]] += 1;
        pairs.append([pair[0],pair[1]]);
    return pairs,vocabulary;


# In[13]:

sentences = []
word_counts = {}
tag_map = {}
vocab=defaultdict(int);
tags = []

# get tag_map, word_counts and cleaned sentences from the training corpus
for line in open('twt.train.json', 'r'):
    cleanLine = clean(json.loads(line));
    pairs,vocab = generateVocab(cleanLine,vocab);
    sentence,word_counts,tag_map,tags = generateTags(pairs,word_counts,tag_map,tags);
    sentences.append(sentence);


# In[14]:

# REMOVE THE WORDS WHICH ARE CONVERTED TO UNK
vocabulary = defaultdict(int);

for key,value in vocab.items():
    if value > 1:
        vocabulary[key] = value;
#print(tags);


# In[15]:

# TOTAL COUNT OF TAGS
total_count = 0;
for tag in tag_map:
    total_count += tag_map[tag];

# SORT THE TAGS
tags = sorted(tags)
print(tags);


# ##### CALCULATE EMISSION PROBABILITIES

# In[16]:

emission_model=emission_prob(word_counts,tag_map);


# ##### CALCULATE BIGRAM TRANSITION PROBABILITIES

# In[17]:

bigram_map,biunigram_map = bigram_count(sentences);
bigram_transition_model = bigram_prob(bigram_map, biunigram_map);


# ##### CALCULATE TRIGRAM TRANSITION PROBABILITIES

# In[18]:

trigram_map,tribigram_map = trigram_count(sentences);
trigram_transition_model = trigram_prob(trigram_map,tribigram_map);


# ##### GENERATE TAGS FROM DEVELOPMENT DATA

# In[19]:

originalTags=[]
sentences=[]
count = 0

for line in open('twt.dev.json', 'r'):
    sentence = clean(json.loads(line));
    tag,line,count = getTestSentence(sentence,vocabulary,count);
    if len(tag) < 3:
        continue
    originalTags.append(tag);
    sentences.append(line);


# ##### CALCULATE ACCURACY ON BIGRAM VITERBI MODEL USING DEVELOPMENT DATA

# In[20]:

bestBiTags=[]
start_time = time.time()
for sentence in sentences:
    bestBiTags.append(bigram_viterbi(emission_model,bigram_transition_model,tags,sentence));
end_time = time.time()
print("Avg Time taken: " + str(round((end_time - start_time)/len(bestBiTags),2)) + " seconds per sentence")
print("Total Time taken: " + str(round(end_time - start_time,2)) + " seconds for " + str(len(bestBiTags)) + 
      " sentences");


# In[21]:

correct_count,total_count=checkTags(bestBiTags,originalTags);

print('ALL CORRECT - ',correct_count,' :: ALL TOTAL - ', total_count);
print('ACCURACY OF BIGRAM HMM - ', round(100 * correct_count/total_count,2),'%');


# ##### CALCULATE ACCURACY ON TRIGRAM VITERBI MODEL USING DEVELOPMENT DATA

# In[22]:

bestTriTags=[]
start_time = time.time()
for sentence in sentences:
    bestTriTags.append(trigram_viterbi(emission_model,trigram_transition_model,tags,sentence));
end_time = time.time()
print("Avg Time taken: " + str(round((end_time - start_time)/len(bestTriTags),2)) + " seconds per sentence")
print("Total Time taken: " + str(round(end_time - start_time,2)) + " seconds for " + str(len(bestTriTags)) + 
      " sentences");


# In[23]:

correct_count,total_count=checkTags(bestTriTags,originalTags);

print('ALL CORRECT - ',correct_count,' :: ALL TOTAL - ', total_count);
print('ACCURACY OF TRIGRAM HMM - ', round(100 * correct_count/total_count,2),'%');


# ##### CALCULATE TRIGRAM SMOOTHING TRANSITION PROBABILITIES

# In[24]:

def smoothing(lamb1,trigram_probs,lamb2,bigram_probs,lamb3,tag_map,total_count):
    smoothing_model = defaultdict(int);
    for tag1 in tag_map:
        for tag2 in tag_map:
            for tag3 in tag_map:
                smoothing_model[(tag1,tag2,tag3)] = (lamb1*trigram_probs[(tag1,tag2,tag3)] + 
                                                     lamb2*bigram_probs[tag3][tag2] + 
                                                     lamb3*(tag_map[tag3]/total_count));
    # <start>,<start>
    for tag in tag_map:
        smoothing_model[('<start>','<start>',tag)] = (lamb1*trigram_probs[('<start>','<start>',tag)] + 
                                                     lamb2*bigram_probs[tag]['<start>'] + 
                                                     lamb3*(tag_map[tag]/total_count));
    # <start>,word & for </stop>
    for tag1 in tag_map:
        for tag2 in tag_map:
            smoothing_model[('<start>',tag1,tag2)] = (lamb1*trigram_probs[('<start>',tag1,tag2)] + 
                                                     lamb2*bigram_probs[tag2][tag1] + 
                                                     lamb3*(tag_map[tag2]/total_count));
            smoothing_model[(tag1,tag2,'</stop>')] = (lamb1*trigram_probs[(tag1,tag2,'</stop>')] + 
                                                     lamb2*bigram_probs['</stop>'][tag2]);
    
    return smoothing_model;
                
    
lamb1 = 0.9;
lamb2 = 0.08;
lamb3 = 0.02;
transitionModel = smoothing(lamb1,trigram_transition_model,lamb2,bigram_transition_model,lamb3,tag_map,total_count);


# ##### CALCULATE ACCURACY ON LINEAR INTERPOLATION TRIGRAM VITERBI MODEL USING DEVELOPMENT DATA

# In[25]:

bestInterTags=[]
start_time = time.time()
for sentence in sentences:
    bestInterTags.append(trigram_viterbi_inter(emission_model,transitionModel,tags,sentence));
end_time = time.time()
print("Avg Time taken: " + str(round((end_time - start_time)/len(bestInterTags),2)) + " seconds per sentence")
print("Total Time taken: " + str(round(end_time - start_time,2)) + " seconds for " + str(len(bestInterTags)) + 
      " sentences");


# In[26]:

correct_count,total_count=checkTags(bestInterTags,originalTags);

print('ALL CORRECT - ',correct_count,' :: ALL TOTAL - ', total_count);
print('ACCURACY OF TRIGRAM HMM - ', round(100 * correct_count/total_count,2),'%');


# ##### PREPARE TEST DATA FOR TESTING HMMs

# In[27]:

originalTestTags=[]
testSentences=[]
count = 0

for line in open('twt.test.json', 'r'):
    sentence = clean(json.loads(line));
    tag,line,count = getTestSentence(sentence,vocabulary,count);
    if len(tag) < 3:
        continue
    originalTestTags.append(tag);
    testSentences.append(line);


# ##### CALCULATE ACCURACY ON BIGRAM VITERBI MODEL USING TEST DATA

# In[28]:

bestBiTestTags=[]
start_time = time.time()
for sentence in testSentences:
    bestBiTestTags.append(bigram_viterbi(emission_model,bigram_transition_model,tags,sentence));
end_time = time.time()
print("Avg Time taken: " + str(round((end_time - start_time)/len(bestBiTestTags),2)) + " seconds per sentence")
print("Total Time taken: " + str(round(end_time - start_time,2)) + " seconds for " + str(len(bestBiTestTags)) + 
      " sentences");


# In[29]:

correct_count,total_count=checkTags(bestBiTestTags,originalTestTags);

print('ALL CORRECT - ',correct_count,' :: ALL TOTAL - ', total_count);
print('ACCURACY OF BRIGRAM HMM ON TEST DATA - ', round(100 * correct_count/total_count,2),'%');


# ###### ERROR ANALYSIS USING CONFUSION MATRIX

# In[30]:

compare_testTags = [];
compare_bestBiTags = [];
for index,testTags in enumerate(originalTestTags):
    for innerIndex,tag in enumerate(testTags):
        compare_testTags.append(tag);
        compare_bestBiTags.append(bestBiTestTags[index][innerIndex]);
print('CONFUSION MATRIX FOR BIGRAM HMM USING TEST DATASET -- \n');
print(confusion_matrix(compare_testTags,compare_bestBiTags,tags));


# ##### CALCULATE ACCURACY ON LINEAR INTERPOLATION TRIGRAM VITERBI MODEL USING TEST DATA

# In[31]:

bestInterTestTags=[]
start_time = time.time()
for sentence in testSentences:
    bestInterTestTags.append(trigram_viterbi_inter(emission_model,transitionModel,tags,sentence));
end_time = time.time()
print("Avg Time taken: " + str(round((end_time - start_time)/len(bestInterTestTags),2)) + " seconds per sentence")
print("Total Time taken: " + str(round(end_time - start_time,2)) + " seconds for " + str(len(bestInterTestTags)) + 
      " sentences");


# In[32]:

correct_count,total_count=checkTags(bestInterTestTags,originalTestTags);

print('ALL CORRECT - ',correct_count,' :: ALL TOTAL - ', total_count);
print('ACCURACY OF LINEAR INTERPOLATION TRIGRAM HMM - ', round(100 * correct_count/total_count,2),'%');


# ###### ERROR ANALYSIS USING CONFUSION MATRIX FOR LINEAR INTERPOLATION TRIGRAM HMM

# In[33]:

compare_testTags = [];
compare_bestTriTags = [];
for index,testTags in enumerate(originalTestTags):
    for innerIndex,tag in enumerate(testTags):
        compare_testTags.append(tag);
        compare_bestTriTags.append(bestInterTestTags[index][innerIndex]);
print('CONFUSION MATRIX FOR LINEAR INTERPOLATION TRIGRAM HMM USING TEST DATASET -- \n');
print(confusion_matrix(compare_testTags,compare_bestTriTags,tags));


# ## EXPECTATION MAXIMIZATION

# #### FORWARD BACKWARD FOR BIGRAM

# In[34]:

def forward_backward_bigram(emissions,transitions,tags,line,bigrams,unigrams,wordCounts):
    
    columns = len(line) + 2;
    rows = len(tags);
    
    forwards = np.zeros((rows, columns));
    backwards = np.zeros((rows, columns));
    
    forwards[:, 0] = 1.0;
    backwards[:, columns-1] = 1.0;
    
    for row,tag in enumerate(tags):
        forwards[row][1] = transitions[tag]['<start>']*emissions[line[0]][tag];
        backwards[row][columns-2] = transitions['</stop>'][tag];
    
    for index in range(2,columns-1):
        for row,tag in enumerate(tags):
            emission_prob = emissions[line[index-1]][tag];
            total_prob = 0.0
            for internalRow,oldTag in enumerate(tags):
                total_prob += transitions[tag][oldTag]*forwards[internalRow][index-1]*emission_prob;
            forwards[row][index] = total_prob;
    
    for index in reversed(range(1,columns-2)):
        for row,tag in enumerate(tags):
            total_prob = 0.0
            for internalRow,newTag in enumerate(tags):
                emission_prob = emissions[line[index]][newTag];
                total_prob += transitions[newTag][tag]*backwards[internalRow][index+1]*emission_prob;
            backwards[row][index] = total_prob;
    
    for row,tag in enumerate(tags):
        forwards[row][columns-1] = transitions['</stop>'][tag]*forwards[row][columns-2];
        backwards[row][0] = transitions[tag]['<start>']*backwards[row][1]*emissions[line[0]][tag];
        
    marginal_probs = forwards*backwards;
    if math.isclose(sum(marginal_probs.sum(0)),0.0):
        return bigrams,unigrams,wordCounts;
    expectations = marginal_probs/marginal_probs.sum(0)
    
    for row,tag in enumerate(tags):
        bigram_start = ('<start>',tag);
        bigrams[bigram_start] += expectations[row][1];
        bigram_end = (tag,'</stop>');
        bigrams[bigram_end] += expectations[row][columns-2];
        unigrams[tag] += expectations[row][1];
        if line[0] not in wordCounts:
            wordCounts[line[0]] = defaultdict(int);
        internal = wordCounts[line[0]];
        internal[tag] += expectations[row][1];
    
    unigrams['<start>'] += 1;
    unigrams['</stop>'] += 1;
    
    for index in range(1,columns-2):
        if line[index] not in wordCounts:
            wordCounts[line[index]] = defaultdict(int);
        internal = wordCounts[line[index]];
        for row,tag in enumerate(tags):
            for nextRow,nextTag in enumerate(tags):
                bigram = (tag,nextTag);
                if bigram not in bigrams:
                    bigrams[bigram] = 0;
                prob = expectations[row][index]*expectations[nextRow][index+1];
                bigrams[bigram] += prob;
            unigrams[tag] += expectations[row][index+1];
            if tag not in internal:
                internal[tag] = 0;
            internal[tag] += expectations[row][index+1];
            
    return bigrams,unigrams,wordCounts;


# #### SPLIT TRAINING DATA INTO SUPERVISED AND UNSUPERVISED

# In[35]:

i = 0;
supervised = []
unsupervised = []

for line in open('twt.train.json', 'r'):
    if i >= 25000:
        unsupervised.append(clean(json.loads(line)));
    else:
        supervised.append(clean(json.loads(line)));
    i = i+1;
print(len(supervised),len(unsupervised));


# ##### TRAIN SUPERVISED USING TRAINING DATA

# In[36]:

super_word_counts = {}
super_tag_map = {}
super_vocab=defaultdict(int);
super_tags = [];
super_sentences = [];

for line in supervised:
    pairs,super_vocab = generateVocab(line,super_vocab);
    sentence,super_word_counts,super_tag_map,super_tags = generateTags(pairs,super_word_counts,super_tag_map,super_tags);
    super_sentences.append(sentence);


# In[37]:

supervised_emission_model=emission_prob(super_word_counts,super_tag_map);


# In[38]:

super_bigram_map,super_biunigram_map = bigram_count(super_sentences);
super_bigram_transition_model = bigram_prob(super_bigram_map, super_biunigram_map);


# In[39]:

# REMOVE THE WORDS WHICH ARE CONVERTED TO UNK
em_vocabulary = defaultdict(int);

for key,value in super_vocab.items():
    if value > 1:
        em_vocabulary[key] = value;


# In[40]:

# TOTAL COUNT OF TAGS
total_count = 0;
for tag in super_tag_map:
    total_count += super_tag_map[tag];
#print(total_count)

# SORT THE TAGS
super_tags = sorted(super_tags)
print(super_tags);


# ##### TESTING SUPERVISED COUNTS FOR EM

# In[41]:

testTags=[]
testSentences=[]
count = 0

for line in unsupervised:
    tag,sentence,count = getTestSentence(line,em_vocabulary,count);
    if len(tag) < 3:
        continue
    testTags.append(tag);
    testSentences.append(sentence);


# ###### ACCURACY CHECK ON BIGRAM VITERBI MODEL

# In[42]:

bestBiTags=[]
start_time = time.time()
for sentence in testSentences:
    bestBiTags.append(bigram_viterbi(supervised_emission_model,super_bigram_transition_model,super_tags,sentence));
end_time = time.time()
print("Avg Time taken: " + str(round((end_time - start_time)/len(bestBiTags),2)) + " seconds per sentence")
print("Total Time taken: " + str(round(end_time - start_time,2)) + " seconds for " + str(len(bestBiTags)) + 
      " sentences");


# In[43]:

correct_count,total_count=checkTags(bestBiTags,testTags);

print('ALL CORRECT - ',correct_count,' :: ALL TOTAL - ', total_count);
print('ACCURACY OF BIGRAM HMM FOR SUPERVISED MODEL - ', round(100 * correct_count/total_count,2),'%');


# ###### RUN FORWARD-BACKWARD FOR BIGRAM UNSUPERVISED MODEL

# In[44]:

un_bigrams=defaultdict(int)
un_unigrams=defaultdict(int)
un_wordCounts={}
start_time = time.time()
for sentence in testSentences:
    un_bigrams,un_unigrams,un_wordCounts=forward_backward_bigram(supervised_emission_model,
                                super_bigram_transition_model,super_tags,sentence,un_bigrams,un_unigrams,un_wordCounts);
end_time = time.time()
print("Total Time taken: " + str(round(end_time - start_time,2)) + " seconds");


# ###### COMBINE EXPECTED COUNTS FROM UNSUPERVISED AND ACTUAL COUNTS FROM SUPERVISED

# In[45]:

combine_word_count = super_word_counts;
combine_tag_map = super_tag_map;
combine_biunigrams = super_biunigram_map;
combine_bigrams = super_bigram_map;

for key in un_wordCounts:
    if key not in combine_word_count:
        combine_word_count[key] = un_wordCounts[key];
        continue;
    for tag in un_wordCounts[key]:
        if tag not in combine_word_count[key]:
            combine_word_count[key][tag] = un_wordCounts[key][tag];
            continue;
        combine_word_count[key][tag] += un_wordCounts[key][tag];

for tag in un_unigrams:
    if not(start_stop in tag for start_stop in ('<start>','</stop>')):
        combine_tag_map[tag] += un_unigrams[tag];
    combine_biunigrams[tag] += un_unigrams[tag];    

for bigram in un_bigrams:
    word1,word2 = bigram;
    if word2 not in combine_bigrams:
        combine_bigrams[word2] = defaultdict(int);
    combine_bigrams[word2][word1] += un_bigrams[bigram];


# ###### TRAIN MODELS AGAIN USING BIGRAM COMBINED COUNTS

# In[46]:

bi_combine_emission_model=emission_prob(combine_word_count,combine_tag_map);


# In[47]:

bi_combine_transition_model = bigram_prob(combine_bigrams, combine_biunigrams);


# ###### TEST SEMI-SUPERVISED MODEL USING BIGRAM VITERBI HMM

# In[48]:

bestEMBiTags=[]
start_time = time.time()
for sentence in testSentences:
    bestEMBiTags.append(bigram_viterbi(bi_combine_emission_model,bi_combine_transition_model,super_tags,sentence));
end_time = time.time()
print("Avg Time taken: " + str(round((end_time - start_time)/len(bestEMBiTags),2)) + " seconds per sentence")
print("Total Time taken: " + str(round(end_time - start_time,2)) + " seconds for " + str(len(bestEMBiTags)) + 
      " sentences");


# In[49]:

correct_count,total_count=checkTags(bestEMBiTags,testTags);

print('ALL CORRECT - ',correct_count,' :: ALL TOTAL - ', total_count);
print('ACCURACY OF BIGRAM HMM - ', round(100 * correct_count/total_count,2),'%');

