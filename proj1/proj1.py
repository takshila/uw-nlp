
# coding: utf-8

# In[1]:

from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
import math
import nltk
import glob


# The replaceUnk method is used to insert unk in the start of every line and the occurence of every word for first time is replaced by unk.

# In[2]:

def replaceUnk(line, first_occ):
    tokenizer = RegexpTokenizer(r'\w+');
    words=tokenizer.tokenize(line);
    newLine = '<unk> ';
    for word in words:
        if word in first_occ:
            newLine += word;
        else:
            newLine += '<unk>';
            first_occ[word] = 1;
        newLine += ' ';
    newLine += '</stop>';
    return newLine.strip(),first_occ;


# The function 'UNIGRAM_COUNTS' takes the data in the array format and returns a dictionary of word counts.

# In[3]:

def unigrams_counts(data):
    word_counts = defaultdict(int);
    total_count = 0;
    first_occ = {};
    for line in data:
        newLine,first_occ=replaceUnk(line,first_occ);
        words = newLine.split(' ');
        for word in words:
            word_counts[word] += 1;
            total_count += 1;
    return word_counts, total_count;


# The function 'BIGRAM_COUNTS' takes the data in the array format and returns a dictionary of word counts.

# In[4]:

def bigrams_counts(data):
    word_counts = defaultdict(int);
    first_occ = {};
    
    for line in data:
        newLine,first_occ=replaceUnk(line, first_occ);
        words = newLine.split(' ');
        bigrams = tuple(nltk.bigrams(words));
        for word in bigrams:
            word_counts[word] += 1;
            
    return word_counts;


# The function 'TRIGRAM_COUNTS' takes the data in the array format and returns a dictionary of word counts.

# In[5]:

def trigrams_counts(data):
    word_counts = defaultdict(int);
    first_occ = {};
    
    for line in data:
        newLine,first_occ=replaceUnk(line,first_occ);
        words=newLine.split(' ');
        trigrams = tuple(nltk.trigrams(words));
        for word in trigrams:
            word_counts[word] += 1;
    
    return word_counts;


# The function readFileData reads the data and returns the list of lines

# In[6]:

def readFileData(filein):
    f = open(filein,'r');
    filedata = f.readlines();
    f.close();
    return filedata;


# Read the data from training file and get the counts of words for unigram,bigram and trigram maps. Also the total word count is returned.

# In[7]:

trainData=readFileData('brown.train.txt');

unigram_map, total_wordcount=unigrams_counts(trainData);
bigram_map=bigrams_counts(trainData);
trigram_map=trigrams_counts(trainData);


# The function 'UNIGRAM_PROBS' takes in the counts of words for unigram maps and returns the probability of unigrams.

# In[8]:

def unigram_probs(unigram_map, total_count):
    probs = {};
    for word in unigram_map:
        probs[word]=unigram_map[word]/total_count;
    return probs;

unigram_model=unigram_probs(unigram_map,total_wordcount);


# The function 'BIGRAM_PROBS' takes in the counts of words for bigram maps and returns the probability of bigrams.

# In[9]:

def bigram_probs(bigram_map, unigram_map):
    probs = {};
    for word in bigram_map:
        word1, word2 = word;
        if unigram_map[word1] == 0:
            print(word1);
        probs[word]=bigram_map[word]/unigram_map[word1];
    return probs;

bigram_model=bigram_probs(bigram_map,unigram_map);


# The function 'TRIGRAM_PROBS' takes in the counts of words for trigram maps and returns the probability of trigrams.

# In[10]:

def trigram_probs(trigram_map, bigram_map):
    probs = {};
    for word in trigram_map:
        word1,word2,word3 = word;
        probs[word]=trigram_map[word]/bigram_map[(word1,word2)];
    return probs;

trigram_model=trigram_probs(trigram_map, bigram_map);


# The function 'replaceTestUnk' replaces the unseen words in the test corpus as 'unk'.

# In[11]:

def replaceTestUnk(line, vocab):
    tokenizer = RegexpTokenizer(r'\w+');
    words=tokenizer.tokenize(line);
    newLine = '<unk> ';
    for word in words:
        if word in vocab:
            newLine += word;
        else:
            newLine += '<unk>';
        newLine += ' ';
    newLine += '</stop>'
    return newLine.strip();


# The function 'EvaluateUNIGRAM' evaluates the unigram model on the unseen test data.

# In[12]:

def evaluateUnigram(testData,unigram_model):
    tokenizer = RegexpTokenizer(r'\w+');
    total_prob=0;
    word_count=0;
    
    for line in testData:
        prob = 0;
        words = tokenizer.tokenize(line);
        word_count += len(words);
        for word in words:
            if word in unigram_model:
                prob += math.log(unigram_model[word],2);
            else:
                prob += math.log(unigram_model['<unk>'],2);
        
        total_prob += prob;
    
    return math.pow(2,(-1/word_count) * total_prob)


# The function 'EvaluateBIGRAM' evaluates the bigram model on the unseen test data.

# In[13]:

def evaluateBigram(testData,bigram_model,vocab):
    total_prob=0;
    word_count=0;
    
    for line in testData:
        prob = 0;
        newLine = replaceTestUnk(line,vocab);
        words = newLine.split(' ');
        word_count += len(words);
        bigrams = tuple(nltk.bigrams(words));
        
        for word in bigrams:
            word1,word2 = word;
            if word in bigram_model:
                prob += math.log(bigram_model[word], 2);
            elif ('<unk>',word2) in bigram_model:
                prob += math.log(bigram_model[('<unk>',word2)],2);
            else:
                prob += math.log(bigram_model[('<unk>','<unk>')],2);
        
        total_prob += prob;
    return math.pow(2,(-1/word_count) * total_prob);


# The function 'EvaluateTRIGRAM' evaluates the trigram model on the unseen test data.

# In[14]:

def evaluateTrigram(testData,trigram_model,bigram_model,vocab):
    total_prob=0;
    word_count=0;
    
    for line in testData:
        prob = 0;
        newLine = replaceTestUnk(line,vocab);
        words = newLine.split(' ');
        word_count += len(words);
        trigrams = tuple(nltk.trigrams(words));
        
        if (words[0],words[1]) in bigram_model:
            prob += math.log(bigram_model[(words[0],words[1])],2);
        else:
            prob += math.log(bigram_model[('<unk>','<unk>')],2);
        
        for word in trigrams:
            word1,word2,word3 = word;
            if word in trigram_model:
                prob += math.log(trigram_model[word],2);
            elif ('<unk>',word2,word3) in trigram_model:
                prob += math.log(trigram_model[('<unk>',word2,word3)],2);
            elif ('<unk>','<unk>',word3) in trigram_model:
                prob += math.log(trigram_model[('<unk>','<unk>',word3)],2);
            else:
                prob += math.log(trigram_model[('<unk>','<unk>','<unk>')],2);
        total_prob += prob;
    return math.pow(2,(-1/word_count) * total_prob);


# We read the development data to evaluate our model and for hyper-tuning parameters in k-smoothing and linear interpolation.

# In[15]:

devData=readFileData('brown.dev.txt');


# Evaluate the MODELS on training and development data.

# In[16]:

print(evaluateUnigram(trainData, unigram_model));
print(evaluateUnigram(devData, unigram_model));


# In[17]:

print(evaluateBigram(trainData,bigram_model,unigram_map));
print(evaluateBigram(devData,bigram_model,unigram_map));


# In[18]:

print(evaluateTrigram(trainData, trigram_model, bigram_model,unigram_map));
print(evaluateTrigram(devData, trigram_model, bigram_model,unigram_map));


# In[19]:

testData=readFileData('brown.test.txt');
print("UNIGRAM TEST -- ",evaluateUnigram(testData, unigram_model))
print("BIGRAM TEST -- ",evaluateBigram(testData, bigram_model, unigram_map))
print("TRIGRAM TEST -- ",evaluateTrigram(testData, trigram_model, bigram_model, unigram_map))


# ### Add-K SMOOTHING

# In[20]:

def kSmoothing_Trigram(testData,k,trigrams_map,bigrams_map,unigrams_map):
    total_prob=0;
    word_count=0;
    
    V = len(unigrams_map);
    
    for line in testData:
        prob = 0;
        #word_count += len(line.split());
        newLine = replaceTestUnk(line,unigrams_map);
        words=newLine.split(' ');
        word_count += len(words);
        trigrams = tuple(nltk.trigrams(words));
        
        bi_count = bigrams_map[(words[0],words[1])];
        uni_count = unigrams_map['<unk>'];
        
        prob_k_smooth = (k+bi_count)/(k*V + uni_count);
        prob += math.log(prob_k_smooth,2);
        
        for word in trigrams:
            prob_k_smooth = 0;
            word1,word2,word3=word;
            bigram = (word1,word2);
            tri_count = trigrams_map[word];
            bi_count = bigrams_map[bigram];
            
            if bi_count != 0:
                prob_k_smooth = (k+tri_count)/(k*V + bi_count);
            else:
                bi_count = bigrams_map[(word2,word3)];
                uni_count = unigrams_map[word2];
                prob_k_smooth = (k+bi_count)/(k*V + uni_count);
            prob += math.log(prob_k_smooth,2);
            
        total_prob += prob;
    
    return math.pow(2,(-1/word_count) * total_prob);


# In[21]:

print(kSmoothing_Trigram(trainData,10,trigram_map,bigram_map,unigram_map));
print(kSmoothing_Trigram(trainData,1,trigram_map,bigram_map,unigram_map));
print(kSmoothing_Trigram(trainData,0.1,trigram_map,bigram_map,unigram_map));
print(kSmoothing_Trigram(trainData,0.01,trigram_map,bigram_map,unigram_map));
print(kSmoothing_Trigram(trainData,0.001,trigram_map,bigram_map,unigram_map));


# In[22]:

print(kSmoothing_Trigram(devData,10,trigram_map,bigram_map,unigram_map));
print(kSmoothing_Trigram(devData,1,trigram_map,bigram_map,unigram_map));
print(kSmoothing_Trigram(devData,0.1,trigram_map,bigram_map,unigram_map));
print(kSmoothing_Trigram(devData,0.01,trigram_map,bigram_map,unigram_map));
print(kSmoothing_Trigram(devData,0.001,trigram_map,bigram_map,unigram_map));


# In[23]:

print("TEST DATA k-SMOOTHING -- ", kSmoothing_Trigram(testData,0.001,trigram_map,bigram_map,
                                                      unigram_map));


# ### LINEAR INTERPOLATION

# In[24]:

def linear_inter(testData,lamb1,trigrams_map,lamb2,bigrams_map,lamb3,unigrams_map,total_count):
    tokenizer = RegexpTokenizer(r'\w+');
    total_prob=0;
    word_count=0;
    
    for line in testData:
        prob = 0;
        #word_count += len(line.split())
        newLine = replaceTestUnk(line,unigrams_map);
        words=newLine.split(' ');
        word_count += len(words);
        trigrams = tuple(nltk.trigrams(words));
        
        bi_count = bigrams_map[(words[0],words[1])];
        uni_count = unigrams_map[words[1]];
        
        prob_lamb = lamb2*(bi_count/unigrams_map['<unk>']);
        prob_lamb += lamb1 * uni_count;
        
        prob += math.log(prob_lamb,2);
        
        for word in trigrams:
            prob_lamb = 0;
            word1,word2,word3=word;
            bigram = (word1,word2);
            tri_count = trigrams_map[word];
            bi_count = bigrams_map[bigram];
            uni_count = unigrams_map[word2];
            
            if bi_count != 0:
                prob_lamb += lamb1*(tri_count/bi_count);
            
            bi_count = bigrams_map[(word2,word3)];
            prob_lamb += lamb2*(bi_count/uni_count);
            
            uni_count = unigrams_map[word3];
            prob_lamb += lamb3*(uni_count/total_count);
            
            prob += math.log(prob_lamb,2);
        total_prob += prob;
        
    return math.pow(2,(-1/word_count) * total_prob);


# In[25]:

lamb1 = 0.6;
lamb2 = 0.3;
lamb3 = 0.1;

print(linear_inter(trainData,lamb1,trigram_map,lamb2,bigram_map,lamb3,unigram_map,
                   total_wordcount));


# In[26]:

lamb1 = 0.4; lamb2 = 0.3; lamb3 = 0.3;
print(linear_inter(devData,lamb1,trigram_map,lamb2,bigram_map,lamb3,unigram_map,
                   total_wordcount));


# In[27]:

lamb1 = 0.4; lamb2 = 0.4; lamb3 = 0.2;
print(linear_inter(devData,lamb1,trigram_map,lamb2,bigram_map,lamb3,unigram_map,
                   total_wordcount));


# In[28]:

lamb1 = 0.5; lamb2 = 0.3; lamb3 = 0.2;
print(linear_inter(devData,lamb1,trigram_map,lamb2,bigram_map,lamb3,unigram_map,
                   total_wordcount));


# In[29]:

lamb1 = 0.6; lamb2 = 0.3; lamb3 = 0.1;
print(linear_inter(devData,lamb1,trigram_map,lamb2,bigram_map,lamb3,unigram_map,
                   total_wordcount));


# In[30]:

lamb1 = 0.7; lamb2 = 0.2; lamb3 = 0.1;
print(linear_inter(devData,lamb1,trigram_map,lamb2,bigram_map,lamb3,unigram_map,
                   total_wordcount));


# In[31]:

lamb1 = 0.4; lamb2 = 0.3; lamb3 = 0.3;
print("TEST DATA LINEAR INTERPOLATION -- ",linear_inter(testData,lamb1,trigram_map,lamb2,
                                                bigram_map,lamb3,unigram_map,total_wordcount));


# ### AUTHORSHIP ATTRIBUTION

# In[32]:

hgwells_train=readFileData('authorship_attribution/train/herbert_george_wells.train.txt');
jausten_train=readFileData('authorship_attribution/train/jane_austen.train.txt');
mtwain_train=readFileData('authorship_attribution/train/mark_twain.train.txt');
conan_train=readFileData('authorship_attribution/train/sir_arthur_conan_doyle.train.txt');
vwoolf_train=readFileData('authorship_attribution/train/virginia_woolf.train.txt');


# GET THE COUNTS OF UNIGRAMS, BIGRAMS, TRIGRAMS FOR THE AUTHORSHIP DATA

# In[33]:

def getTrainingCounts(trainData):
    unigram_train_cnt, total_wordcnt=unigrams_counts(trainData);
    bigram_train_cnt=bigrams_counts(trainData);
    trigram_train_cnt=trigrams_counts(trainData);
    
    return total_wordcount,unigram_train_cnt,bigram_train_cnt,trigram_train_cnt;

hgwells_wordcnt,hgwells_vocab,hgwells_bicnt,hgwells_tricnt=getTrainingCounts(hgwells_train);
jausten_wordcnt,jausten_vocab,jausten_bicnt,jausten_tricnt=getTrainingCounts(jausten_train);
mtwain_wordcnt,mtwain_vocab,mtwain_bicnt,mtwain_tricnt=getTrainingCounts(mtwain_train);
conan_wordcnt,conan_vocab,conan_bicnt,conan_tricnt=getTrainingCounts(conan_train);
vwoolf_wordcnt,vwoolf_vocab,vwoolf_bicnt,vwoolf_tricnt=getTrainingCounts(vwoolf_train);


# GENERATE THE UNIGRAM, BIGRAM, TRIGRAM MODELS 

# In[34]:

def createModels(wordcnt,vocab,bigramcnt,trigramcnt):
    unigram_mod=unigram_probs(vocab,wordcnt);
    bigram_mod=bigram_probs(bigramcnt,vocab);
    trigram_mod=trigram_probs(trigramcnt,bigramcnt);
    
    return unigram_mod,bigram_mod,trigram_mod;


# GENERATE MODELS FOR DIFFERENT AUTHORS

# In[35]:

hgwells_uni,hgwells_bi,hgwells_tri=createModels(hgwells_wordcnt,hgwells_vocab,hgwells_bicnt,
                                                hgwells_tricnt);


# In[36]:

jausten_uni,jausten_bi,jausten_tri=createModels(jausten_wordcnt,jausten_vocab,jausten_bicnt,
                                                jausten_tricnt);


# In[37]:

mtwain_uni,mtwain_bi,mtwain_tri=createModels(mtwain_wordcnt,mtwain_vocab,mtwain_bicnt,
                                             mtwain_tricnt);


# In[38]:

conan_uni,conan_bi,conan_tri=createModels(conan_wordcnt,conan_vocab,conan_bicnt,conan_tricnt);


# In[39]:

vwoolf_uni,vwoolf_bi,vwoolf_tri=createModels(vwoolf_wordcnt,vwoolf_vocab,vwoolf_bicnt,
                                             vwoolf_tricnt);


# METHODS FOR CHECKING PERPLEXITY FOR UNIGRAM, BIGRAM and TRIGRAM MODELS

# In[40]:

def checkUniAutor(testdata,model,per):
    modelPer=evaluateUnigram(testdata,model);
    status=False;
    if per > modelPer:
        per = modelPer;
        status=True;
    return status,per;

def checkBiAutor(testdata,model,vocab,per):
    modelPer=evaluateBigram(testdata,model,vocab);
    status=False;
    if per > modelPer:
        per = modelPer;
        status=True;
    return status,per;

def checkTriAutor(testdata,trimodel,bimodel,vocab,per):
    modelPer=evaluateTrigram(testdata,trimodel,bimodel,vocab);
    status=False;
    if per > modelPer:
        per = modelPer;
        status=True;
    return status,per;


# CHECK THE UNIGRAM MODEL FOR DEVELOPMENT DATA AND CLASSIFY THE DOCUMENTS

# In[41]:

for filename in glob.glob('authorship_attribution/dev/*_0*.txt'):
    devData=readFileData(filename);
    
    uni_per = 0.0;
    status = False;
    
    uni_per=evaluateUnigram(devData,hgwells_uni);
    name = 'herbert_george_wells';
    
    status,uni_per=checkUniAutor(devData,jausten_uni,uni_per);
    if status == True:
        name='jane_austen';
    
    status,uni_per=checkUniAutor(devData,mtwain_uni,uni_per);
    if status == True:
        name='mark_twain';
    
    status,uni_per=checkUniAutor(devData,conan_uni,uni_per);
    if status == True:
        name='sir_arthur_conan_doyle';
        
    status,uni_per=checkUniAutor(devData,vwoolf_uni,uni_per);
    if status == True:
        name='virginia_woolf';
    
    print(filename,name);


# CHECK THE BIGRAM MODEL FOR DEVELOPMENT DATA AND CLASSIFY THE DOCUMENTS

# In[42]:

for filename in glob.glob('authorship_attribution/dev/*_0*.txt'):
    devData=readFileData(filename);
    
    bi_per = 0.0;
    status = False;
    
    bi_per=evaluateBigram(devData,hgwells_bi,hgwells_vocab);
    name = 'herbert_george_wells';
    
    status,bi_per=checkBiAutor(devData,jausten_bi,jausten_vocab,bi_per);
    if status == True:
        name='jane_austen';
    
    status,bi_per=checkBiAutor(devData,mtwain_bi,mtwain_vocab,bi_per);
    if status == True:
        name='mark_twain';
    
    status,bi_per=checkBiAutor(devData,conan_bi,conan_vocab,bi_per);
    if status == True:
        name='sir_arthur_conan_doyle';
        
    status,bi_per=checkBiAutor(devData,vwoolf_bi,vwoolf_vocab,bi_per);
    if status == True:
        name='virginia_woolf';
    
    print(filename,name);


# CHECK THE TRIGRAM MODEL FOR DEVELOPMENT DATA AND CLASSIFY THE DOCUMENTS

# In[43]:

for filename in glob.glob('authorship_attribution/dev/*_0*.txt'):
    devData=readFileData(filename);
    
    tri_per = 0.0;
    status = False;
    
    tri_per=evaluateTrigram(devData,hgwells_tri,hgwells_bi,hgwells_vocab);
    name = 'herbert_george_wells';
    
    status,tri_per=checkTriAutor(devData,jausten_tri,jausten_bi,jausten_vocab,tri_per);
    if status == True:
        name='jane_austen';
    
    status,tri_per=checkTriAutor(devData,mtwain_tri,mtwain_bi,mtwain_vocab,tri_per);
    if status == True:
        name='mark_twain';
    
    status,tri_per=checkTriAutor(devData,conan_tri,conan_bi,conan_vocab,tri_per);
    if status == True:
        name='sir_arthur_conan_doyle';
        
    status,tri_per=checkTriAutor(devData,vwoolf_tri,vwoolf_bi,vwoolf_vocab,tri_per);
    if status == True:
        name='virginia_woolf';
    
    print(filename,name);


# CHECK THE LINEAR INTERPOLATION MODEL FOR DEVELOPMENT DATA AND CLASSIFY THE DOCUMENTS

# In[47]:

def checkLinearInterAutor(testdata,lamb1,trigram_map,lamb2,bigram_map,lamb3,unigram_map,
                   total_wordcount,per):
    modelPer=linear_inter(testdata,lamb1,trigram_map,lamb2,bigram_map,lamb3,unigram_map,
                           total_wordcount);
    status=False;
    if per > modelPer:
        per=modelPer;
        status=True;
    return status,per;


# In[64]:

lamb1 = 0.55; lamb2 = 0.40; lamb3 = 0.05;

for filename in glob.glob('authorship_attribution/dev/*_0*.txt'):
    devData=readFileData(filename);
    
    li_per = 0.0;
    status = False;
    
    li_per = linear_inter(devData,lamb1,hgwells_tricnt,lamb2,hgwells_bicnt,lamb3,
                          hgwells_vocab,hgwells_wordcnt);
    name = 'herbert_george_wells';
    
    status,li_per=checkLinearInterAutor(devData,lamb1,jausten_tricnt,lamb2,jausten_bicnt,lamb3,
                                        jausten_vocab,jausten_wordcnt,li_per);
    if status == True:
        name='jane_austen';
    
    status,li_per=checkLinearInterAutor(devData,lamb1,mtwain_tricnt,lamb2,mtwain_bicnt,lamb3,
                                        mtwain_vocab,mtwain_wordcnt,li_per);
    if status == True:
        name='mark_twain';
    
    status,li_per=checkLinearInterAutor(devData,lamb1,conan_tricnt,lamb2,conan_bicnt,lamb3,
                                        conan_vocab,conan_wordcnt,li_per);
    if status == True:
        name='sir_arthur_conan_doyle';
        
    status,li_per=checkLinearInterAutor(devData,lamb1,vwoolf_tricnt,lamb2,vwoolf_bicnt,lamb3,
                                        vwoolf_vocab,vwoolf_wordcnt,li_per);
    if status == True:
        name='virginia_woolf';
    
    print(filename,name);


# #### CHECK THE UNIGRAM MODEL FOR TEST DATA AND CLASSIFY THE DOCUMENTS

# In[44]:

for filename in glob.glob('authorship_attribution/test/*_0*.txt'):
    devData=readFileData(filename);
    
    uni_per = 0.0;
    
    uni_per=evaluateUnigram(devData,hgwells_uni);
    name = 'herbert_george_wells';
    
    status,uni_per=checkUniAutor(devData,jausten_uni,uni_per);
    if status == True:
        name='jane_austen';
    
    status,uni_per=checkUniAutor(devData,mtwain_uni,uni_per);
    if status == True:
        name='mark_twain';
    
    status,uni_per=checkUniAutor(devData,conan_uni,uni_per);
    if status == True:
        name='sir_arthur_conan_doyle';
        
    status,uni_per=checkUniAutor(devData,vwoolf_uni,uni_per);
    if status == True:
        name='virginia_woolf';
    
    print(filename,name);


# #### CHECK THE BIGRAM MODEL FOR TEST DATA AND CLASSIFY THE DOCUMENTS

# In[45]:

for filename in glob.glob('authorship_attribution/test/*_0*.txt'):
    devData=readFileData(filename);
    
    bi_per = 0.0;
    
    bi_per=evaluateBigram(devData,hgwells_bi,hgwells_vocab);
    name = 'herbert_george_wells';
    
    status,bi_per=checkBiAutor(devData,jausten_bi,jausten_vocab,bi_per);
    if status == True:
        name='jane_austen';
    
    status,bi_per=checkBiAutor(devData,mtwain_bi,mtwain_vocab,bi_per);
    if status == True:
        name='mark_twain';
    
    status,bi_per=checkBiAutor(devData,conan_bi,conan_vocab,bi_per);
    if status == True:
        name='sir_arthur_conan_doyle';
        
    status,bi_per=checkBiAutor(devData,vwoolf_bi,vwoolf_vocab,bi_per);
    if status == True:
        name='virginia_woolf';
    
    print(filename,name);


# #### CHECK THE TRIGRAM MODEL FOR TEST DATA AND CLASSIFY THE DOCUMENTS

# In[46]:

for filename in glob.glob('authorship_attribution/test/*_0*.txt'):
    devData=readFileData(filename);
    
    bi_per = 0.0;
    
    bi_per=evaluateTrigram(devData,hgwells_tri,hgwells_bi,hgwells_vocab);
    name = 'herbert_george_wells';
    
    status,bi_per=checkTriAutor(devData,jausten_tri,jausten_bi,jausten_vocab,bi_per);
    if status == True:
        name='jane_austen';
    
    status,bi_per=checkTriAutor(devData,mtwain_tri,mtwain_bi,mtwain_vocab,bi_per);
    if status == True:
        name='mark_twain';
    
    status,bi_per=checkTriAutor(devData,conan_tri,conan_bi,conan_vocab,bi_per);
    if status == True:
        name='sir_arthur_conan_doyle';
        
    status,bi_per=checkTriAutor(devData,vwoolf_tri,vwoolf_bi,vwoolf_vocab,bi_per);
    if status == True:
        name='virginia_woolf';
    
    print(filename,name);


# #### CHECK THE LINEAR INTERPOLATION MODEL FOR TEST DATA AND CLASSIFY THE DOCUMENTS

# In[63]:

lamb1 = 0.55; lamb2 = 0.40; lamb3 = 0.05;

for filename in glob.glob('authorship_attribution/test/*_0*.txt'):
    devData=readFileData(filename);
    
    li_per = 0.0;
    status = False;
    
    li_per = linear_inter(devData,lamb1,hgwells_tricnt,lamb2,hgwells_bicnt,lamb3,
                          hgwells_vocab,hgwells_wordcnt);
    name = 'herbert_george_wells';
    
    status,li_per=checkLinearInterAutor(devData,lamb1,jausten_tricnt,lamb2,jausten_bicnt,lamb3,
                                        jausten_vocab,jausten_wordcnt,li_per);
    if status == True:
        name='jane_austen';
    
    status,li_per=checkLinearInterAutor(devData,lamb1,mtwain_tricnt,lamb2,mtwain_bicnt,lamb3,
                                        mtwain_vocab,mtwain_wordcnt,li_per);
    if status == True:
        name='mark_twain';
    
    status,li_per=checkLinearInterAutor(devData,lamb1,conan_tricnt,lamb2,conan_bicnt,lamb3,
                                        conan_vocab,conan_wordcnt,li_per);
    if status == True:
        name='sir_arthur_conan_doyle';
        
    status,li_per=checkLinearInterAutor(devData,lamb1,vwoolf_tricnt,lamb2,vwoolf_bicnt,lamb3,
                                        vwoolf_vocab,vwoolf_wordcnt,li_per);
    if status == True:
        name='virginia_woolf';
    
    print(filename,name);