#!/usr/bin/env python3

import sys, os
import numpy as np
import pandas as pd
from IPython import embed
from collections import Counter
from nltk import TweetTokenizer
import tensorflow as tf
from gensim.models import Word2Vec
import pickle, gzip

class DataReader(object):
  """Reads in the data, and extracts the features (depending on what we need them to be)"""
  def __init__(self, train_path, test_path,
               text_field="text",sent_field="sent_bin",
               vocab_size=10000):
    """Train_path contains the training data (general tweets)
    test_path contains the location of the test data (android iphone tweets)"""
    self.train_data = pd.read_pickle(train_path)
    self.test_data = pd.read_pickle(test_path)
    self.txt_field = text_field
    self.sent_field = sent_field
    self.vocab_size = vocab_size
    self.tok = TweetTokenizer()

  def tokenize_str(self,s):
    """Tokenizes a string"""
    return [w.lower() for w in self.tok.tokenize(s)]

  def make_one_hot_vocab(self):
    """Turns words into integers."""
    ### Vocab
    words = [w for tweet in self.train_data[self.txt_field] for w in self.tokenize_str(tweet)]
    c = Counter(words)
    counts = c.most_common(self.vocab_size-1)
    counts.insert(0,("OOV",0))
    self.tok_to_id = Counter({w[0]:i for w,i in zip(counts,range(self.vocab_size))})
    self.train_data["tok_"+self.txt_field] = [[self.tok_to_id[w] for w in self.tokenize_str(t)]
                                              for t in self.train_data[self.txt_field]]
    self.test_data["tok_"+self.txt_field] = [[self.tok_to_id[w] for w in self.tokenize_str(t)]
                                              for t in self.test_data[self.txt_field]]
    
    self.max_seq_len = max(map(len,self.train_data["tok_"+self.txt_field]))
    self.max_seq_len = max(self.max_seq_len,
                           max(map(len,self.test_data["tok_"+self.txt_field])))

  def make_vector_vocab(self, vectors):
    """Uses the existing w2v from Google News to create the tokenized data"""
    words = [w for tweet in self.train_data[self.txt_field] for w in self.tokenize_str(tweet)]
    voc = set(words) & vectors.keys()
    self.tok_to_id = {k: np.concatenate([vectors[k],[0]]) for k,v in vectors.items() if k in voc}
    
    self.vocab_size = len(self.tok_to_id)
    self.feat_dims = list(self.tok_to_id.values())[0].shape[0]
    # We add an extra row (301th) for OOV mentions, so we don't remove them
    oov = np.concatenate([np.zeros(self.feat_dims-1),[1]])
    self.train_data["tok_"+self.txt_field] = [[self.tok_to_id.get(w,oov) for w in self.tokenize_str(t)]
                                              for t in self.train_data[self.txt_field]]
    self.test_data["tok_"+self.txt_field] = [[self.tok_to_id.get(w,oov) for w in self.tokenize_str(t)]
                                              for t in self.test_data[self.txt_field]]
    
  def yield_vector_batches(self, batch_size, data="training"):
    """Tokenizes tweets, keep the N most frequent tokens of the training data (maps the others to OOV)
    Then breaks up the data into batches for training the model"""
    cols = ["tok_"+self.txt_field,self.sent_field]
    
    if data == "training":
      data = self.train_data
    elif data == "testing" and hasattr(self,"tok_to_id"):
      data = self.test_data
    else:
      raise ValueError("specify either training or testing as data")
    for c in cols:
      data = data[np.logical_not(pd.isnull(data[c]))]

    num_yields = int(np.ceil(len(data)/batch_size))

    for i in range(num_yields):
      chunk = data[i*batch_size:(i+1)*batch_size]
      x = np.array([np.average(s,axis=0) for s in chunk["tok_"+self.txt_field]])
      y = chunk[self.sent_field]# .as_matrix()
      yield x,y
      
  def yield_one_hot_batches(self, batch_size, data="training"):
    """Tokenizes tweets, keep the N most frequent tokens of the training data (maps the others to OOV)
    Then breaks up the data into batches for training the model"""
    cols = ["tok_"+self.txt_field,self.sent_field]
    if data == "training":
      data = self.train_data
    elif data == "testing" and hasattr(self,"tok_to_id"):
      data = self.test_data
    else:
      raise ValueError("specify either training or testing as data")
    for c in cols:
      data = data[np.logical_not(pd.isnull(data[c]))]
    

    num_yields = int(np.ceil(len(data)/batch_size))

    for i in range(num_yields):
      
      chunk = data[i*batch_size:(i+1)*batch_size]

      dfs = []
      for x in chunk["tok_"+self.txt_field]:
        d = {k:v/len(x) for k,v in Counter(x).items()}
        s = pd.Series(data=d,index=list(range(self.vocab_size)))
        s.fillna(0,inplace=True)
        dfs.append(s)
        
      x = pd.concat(dfs,axis=1).T.as_matrix()
      y = chunk[self.sent_field] #.as_matrix()
      
      yield x,y
    
class Classifier(object):
  """This is the prediction model. Only does binary classification"""
  def __init__(self,num_feats,hidden_size,is_training=True):
    # Define tensorflow graph and session. All variables and operations are tied to this graph
    self.graph = graph = tf.Graph()
    self.session = session = tf.Session(graph=self.graph)
    self.num_feats = num_feats
    self.hidden_size = hidden_size
    
    with graph.as_default():
      # Define input feature placeholder
      # our data is the float type
      # we can give it minibatches of any size, so the 1st dim is None (allows for flexibility)
      # second dimension is the number of features considered (one hot vocab size or vector dim)
      self._feats = tf.placeholder(tf.float32,
                                   [None,num_feats],
                                   "feats")

      # Define target label placeholder
      # Same as before, None is to allow for various mini batch sizes.
      self._target_label = tf.placeholder(tf.int32,
                                          [None,],
                                          "target_label")

      
      ### <CSE490>
      ### Fill this in here (add a bias, add a non-linearity, extra layer)


      # Weights -> transform your space into a lower-dimensional vector (here:2)
      # get variable checks for a variable named w, if exists, uses it
      # if it doesn't, initializes it (usually doesn't exist, but it's
      # easier to not have to explicitly give an initial value)
      #w = tf.get_variable("w",
      #                    [num_feats,2], # if you add a layer, change this
      #                    tf.float32)
                          
      # define ("get") more weight matrices and biases here
      
      # h: un-normalized probabilities (change this)
      # Don't rename h, it's used below
      #h = tf.nn.xw_plus_b(self._feats, w,tf.zeros([2]))
      
      
      #### FEED-FORWARD NEURAL NETWORK
      w1 = tf.get_variable("w1", [num_feats,hidden_size], tf.float32)
      h1 = tf.tanh(tf.nn.xw_plus_b(self._feats,w1,tf.ones([hidden_size])))
      
      w2 = tf.get_variable("w2", [hidden_size,2], tf.float32)
      h = tf.sigmoid(tf.nn.xw_plus_b(h1, w2, tf.ones([2])))
      
      ### </CSE490>
      # Make it a probability distribution
      self._softmaxes = tf.nn.softmax(h)

      # Pick class
      self._pred_label = tf.argmax(self._softmaxes,1)

      # Loss calculated as to make the prob. dist. closer to target
      self._loss = loss = tf.nn.sparse_softmax_cross_entropy_with_logits(h,self._target_label)
      
      self._cost = tf.reduce_mean(loss)
      
      if is_training:
        # No kinks optimizer
        optimizer = tf.train.AdamOptimizer()
        self._train_op = optimizer.minimize(loss)

        self.session.run(tf.initialize_all_variables())
      
  @property
  def train_op(self):
    return self._train_op

  @property
  def softmaxes(self):
    return self._softmaxes

  @property
  def cost(self):
    return self._cost

  @property
  def loss(self):
    return self._loss

  @property
  def feats(self):
    return self._feats

  @property
  def pred_label(self):
    return self._pred_label

  @property
  def target_label(self):
    return self._target_label

def save_model(path,reader,model):
  with open(path+".pkl","wb+") as f:
    pickle.dump(reader,f)
  with model.graph.as_default():
    saver = tf.train.Saver()
  ckpt = os.path.basename(path+".ckpt")
  saver.save(model.session, path+".tf", latest_filename=ckpt)
  
#################################################
#################################################
#################################################
### <CSE490>
# Play around with this if you want.
#  * EPOCHS: number of passes through the training data.
#  * Mini batch size: number of tweets to use for one training iteration.
#  * One hot vocab size: size of the vocabulary, only relevant when using one-hot representation.
#  * Hidden size: dimension of the hidden layer (assuming you implemented it :D )

EPOCHS = 10
MINI_BATCH_SIZE = 100
ONE_HOT_VOCAB_SIZE = 10000
HIDDEN_SIZE = 200

### </CSE490>
def prediction_analysis(preds,targets,ids,reader):
  df = pd.DataFrame(data=np.array([preds,targets]).T,index=ids,columns=["pred","target"])
  df['and_iph'] = reader.test_data.ix[df.index,0]
  acc = sum(preds == targets) / len(targets)
  mfc = sum(targets == 0) / len(targets)
  print("Acc: {:.4f}, mfc: {:.4f}".format(acc,mfc))

  and_pct_pos = df[df.and_iph == 0].pred.mean()*100
  iph_pct_pos = df[df.and_iph == 1].pred.mean()*100
  print("Based on the current model")
  print("{:.2f}% of Android tweets are positive".format(and_pct_pos))
  print("{:.2f}% of iPhone tweets are positive".format(iph_pct_pos))


def main(path=None):
  if path is None or path == "one_hot":
    print("ONE HOT")
    r = DataReader("sampleSent.pkl","sourceSent.pkl",vocab_size=ONE_HOT_VOCAB_SIZE)
    r.make_one_hot_vocab()
    m = Classifier(r.vocab_size,HIDDEN_SIZE)
    train_iterator = r.yield_one_hot_batches
    test_iterator = r.yield_one_hot_batches
  else:
    print("Loading {}".format(path))
    with open(path,"rb") as f:
      vectors = pickle.load(f)
    print(path,"... Done",file=sys.stderr)
    r = DataReader("sampleSent.pkl","sourceSent.pkl")
    print("Created reader & loaded data",file=sys.stderr)
  
    r.make_vector_vocab(vectors)
  
    m = Classifier(r.feat_dims,HIDDEN_SIZE)
    train_iterator = r.yield_vector_batches
    test_iterator = r.yield_vector_batches
    
  print("Vocab_size",r.vocab_size)
  for step in range(EPOCHS):
    print("    Starting epoch {}".format(step),file=sys.stderr)

    total_cost = 0
    for i,(x,y) in enumerate(train_iterator(MINI_BATCH_SIZE)):
      cost, op = m.session.run(
        [m.cost,m.train_op],
        {m.feats: x,
         m.target_label: y.as_matrix()
        }
      )
      total_cost += cost
      if i % 10 == 0 and i != 0:
        print("Step {}, cost {}".format(i,cost),file=sys.stderr)
        
  total_preds = [np.array([])]
  total_targets = [np.array([])]
  total_ids = []
  for i,(x,y) in enumerate(test_iterator(MINI_BATCH_SIZE,"testing")):

    cost, preds = m.session.run(
      [m.cost,m.pred_label],
      {m.feats: x,
       m.target_label: y.as_matrix()
      }
    )
    total_preds.append(preds)
    total_targets.append(y.as_matrix())
    total_ids.extend(list(y.index))
  preds = np.concatenate(total_preds)
  targets = np.concatenate(total_targets)
  prediction_analysis(preds,targets,total_ids,r)
  
if __name__ == '__main__':
  main(sys.argv[1])
