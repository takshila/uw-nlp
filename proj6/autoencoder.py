#!/usr/bin/env python3

import sys, os
import numpy as np
import pandas as pd
from IPython import embed
from collections import Counter
from nltk import TweetTokenizer
import tensorflow as tf
import pickle, gzip, json

# Our own RNN code
import rnn_cell

class DataReader(object):
  """Reads in the data, and extracts the features (depending on what we need them to be)"""
  def __init__(self, train_path, test_path,
               vocab_size=10000):
    """Train_path contains the training data (general tweets)
    test_path contains the location of the test data (android iphone tweets)"""
    with gzip.open(train_path,"rt") as f:
      self.train_data = [l for l in json.load(f) if l != " "]
    with gzip.open(test_path,"rt") as f:
      self.test_data = [l for l in json.load(f) if l != " "]

    self.vocab_size = vocab_size
    self.tok = TweetTokenizer()

  def tokenize_str(self,s,eom=False):
    """Tokenizes a string"""
    if eom:
      return ["<BOM>"]+[w.lower() for w in self.tok.tokenize(s)]+["<EOM>"]
    else:
      return [w.lower() for w in self.tok.tokenize(s)]

  def make_one_hot_vocab(self):
    """Turns words into integers."""
    ### Vocab
    words = [w for tweet in self.train_data for w in self.tokenize_str(tweet)]
    c = Counter(words)
    print("{} distinct words in the corpus".format(len(c)))
    counts = c.most_common(self.vocab_size-4)
    counts.insert(0,("<PAD>",0))
    counts.insert(0,("<BOM>",0))
    counts.insert(0,("<EOM>",0))
    counts.insert(0,("<OOV>",0))
    self.tok_to_id = Counter({w[0]:i for w,i in zip(counts,range(self.vocab_size))})
    self.tok_train_data = [[self.tok_to_id[w] for w in self.tokenize_str(t,True)]
                           for t in self.train_data]
    self.tok_test_data = [[self.tok_to_id[w] for w in self.tokenize_str(t,True)]
                          for t in self.test_data]
    
    self.max_seq_len = max(map(len,self.tok_train_data))
    self.max_seq_len = max(self.max_seq_len,
                           max(map(len,self.tok_test_data)))

  def pad(self):
    return self.tok_to_id["<PAD>"]
  
  def eom(self):
    return self.tok_to_id["<EOM>"]
  
  def yield_one_hot_batches(self, batch_size, data_name="training"):
    """Tokenizes tweets, keep the N most frequent tokens of the training data (maps the others to OOV)
    Then breaks up the data into batches for training the model"""
    if data_name == "training":
      data = self.tok_train_data
    elif data_name == "testing" and hasattr(self,"tok_to_id"):
      data = self.tok_test_data
    else:
      raise ValueError("specify either training or testing as data")
    
    np.random.shuffle(data)
    num_yields = int(np.ceil(len(data)/batch_size))

    for i in range(num_yields):
      
      chunk = data[i*batch_size:(i+1)*batch_size]

      dfs = []
      for x in chunk:
        d = {k:v/len(x) for k,v in Counter(x).items()}
        s = pd.Series(data=d,index=list(range(self.vocab_size)))
        s.fillna(0,inplace=True)
        dfs.append(s)
        
      x = pd.concat(dfs,axis=1).T.as_matrix()
      y = chunk
      y_len = np.array([len(i)+1 for i in y])
      y = [i+[self.pad()]*(self.max_seq_len - len(i)) for i in y]
      y = np.array(y)

      assert y.shape[1] == self.max_seq_len
      assert y_len.shape[0] == x.shape[0]
      assert y.shape[0] == x.shape[0]
      
      yield x,y,y_len
      
  def ids_to_str(self,ids,pads=False):
    id_to_tok = {v:k for k,v in self.tok_to_id.items()}
    sentences = []
    for s in ids:
      i = s.index(self.pad()) if self.pad() in s else len(s)
      i = min(i,s.index(self.eom()) if self.eom() in s else len(s))
      sentences.append(" ".join([id_to_tok[d] for d in s[:i]]))
    return sentences
      
class BOW2Seq(object):
  """This is the prediction model. Only does binary classification"""
  def __init__(self,vocab_size,max_seq_len,hidden_size,is_training=True):
    # Define tensorflow graph and session. All variables and operations are tied to this graph
    self.graph = graph = tf.Graph()
    self.session = session = tf.Session(graph=self.graph)
    self.vocab_size = vocab_size
    self.max_seq_len = max_seq_len
    self.hidden_size = hidden_size
    
    with graph.as_default():
      # Define input feature placeholder
      # our data is the float type
      # we can give it minibatches of any size, so the 1st dim is None (allows for flexibility)
      # second dimension is the vocab size
      self._bow = tf.placeholder(tf.float32,
                                 [None,vocab_size],
                                 "bow")

      # Define target sequence placeholder
      # Same as before, None is to allow for various mini batch sizes.
      self._target_seq = tf.placeholder(tf.int64,
                                          [None,max_seq_len],
                                          "target_seq")
      # Define target sequence lengths placeholder
      # Same as before, None is to allow for various mini batch sizes.
      self._target_seq_len = tf.placeholder(tf.int64,
                                          [None,],
                                          "target_seq_len")


      with tf.variable_scope("embedding"):
        # Embed the BOW into a smaller space
        w = tf.get_variable("embed_w",
                            [vocab_size,hidden_size],
                            tf.float32)
        b = tf.get_variable("embed_b",
                            [hidden_size],
                            tf.float32)
        emb_bow = tf.nn.xw_plus_b(self._bow,w,b)

      with tf.variable_scope("unembedding"):
        # We chose to apply a different embedding transformation
        # to the target sequence.
        emb = tf.get_variable("unembed",
                            [vocab_size,hidden_size],
                              tf.float32)
        # Our target sequence is now in the hidden space
        emb_targets = tf.nn.embedding_lookup(emb,self._target_seq)
        emb_target_list = tf.unpack(tf.transpose(emb_targets,[1,0,2]))
      target_list = tf.unpack(tf.transpose(self._target_seq))

      ## <CSE490> RNN cell stuff
      #cell = rnn_cell.BasicLSTMCell(hidden_size)
      #init_state = tf.concat(1,[emb_bow]*2)
      # If GRU, just uncomment below (and comment above?)
      cell = rnn_cell.GRUCell(hidden_size)
      init_state = emb_bow
      ## </CSE490>

      if is_training:
        # When training, we can take the entire sequence and feed it into this ``rnn'' function
        # since we're feeding the target sequence as the input to the decoder.
        emb_output_list, _ = tf.nn.rnn(cell,emb_target_list,initial_state=init_state,sequence_length=self._target_seq_len)

        # We shift the predicted outputs, because at each word we're trying to predict the next
        emb_output_list = emb_output_list[:-1]
        
        with tf.variable_scope("unembedding"):
          # Outputs are still in the ``hidden_size'' dimension
          # we need to make them ``vocab_size''
          un_emb = tf.transpose(emb)
          output_list = [tf.matmul(t,un_emb)
                        for t in emb_output_list]
          outputs = tf.transpose(tf.pack(output_list),[1,0,2])
          
          # making them a probability distribution
          softmax_list = [tf.nn.softmax(t) for t in output_list]
      else:
        # Here, we're no longer training, so at each decoding step, for each word that we
        # chose, we have to pick the most probable word from the prob. dist. given by the
        # softmax. This is why this code loops.
        state = init_state
        # The first symbol is always the "<BOM>" (beginning of message) symbol
        inp = emb_target_list[0]
        with tf.variable_scope("unembedding"):
          # defining the un-embedding parameter.
          un_emb = tf.transpose(emb)
        output_list = []
        softmax_list = []
        with tf.variable_scope("RNN"):
          for i in range(max_seq_len):
            if i!=0:
              # needed to call the cell multiple times
              tf.get_variable_scope().reuse_variables()

            # Take one pass through the cell
            emb_out, state = cell(inp,state)

            # Creating prob. dist. over next symbol 
            out = tf.matmul(emb_out,un_emb)
            sm = tf.nn.softmax(out)

            # Taking most probable output and embedding it again.
            inp = tf.nn.embedding_lookup(emb,tf.argmax(sm,1))

            # Saving the output distribution
            output_list.append(out)
            softmax_list.append(sm)
            
          # Again, we're shifting the output, see above  
          output_list = output_list[:-1]
          softmax_list = softmax_list[:-1]

      # If we want to query the graph, we can't query a list, we need to
      # make it a tensor.
      self._softmaxes = tf.transpose(tf.pack(softmax_list),[1,0,2])

      # Sequences have variable length, but have been padded to all be length ``max_seq_len''.
      # However, we don't want to compute the cost based on how well we predict the padding
      # symbols, so the cost weights are dynamically set to 0 when we're in the padded zone.
      weights = [tf.select(tf.less_equal(
        self._target_seq_len-2,
        tf.ones_like(self._target_seq_len)*i),tf.zeros_like(self._target_seq_len,tf.float32),tf.ones_like(self._target_seq_len,tf.float32))
              for i in range(max_seq_len)]
      weights = weights[:-1]
      self._weights = tf.transpose(tf.pack(weights))

      # Computes the cross-entropy
      self._loss = loss = tf.nn.seq2seq.sequence_loss_by_example(output_list,target_list[1:],weights)
      self._cost = cost = tf.reduce_mean(loss)
      
      if is_training:
        # No kinks optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self._train_op = optimizer.minimize(loss)

        self.session.run(tf.initialize_all_variables())
        print("Setting up training")
      print("Created graph, and initialized variables (BOW2Seq)")
      
  @property
  def train_op(self):
    return self._train_op

  @property
  def softmaxes(self):
    return self._softmaxes

  @property
  def weights(self):
    return self._weights

  @property
  def cost(self):
    return self._cost

  @property
  def loss(self):
    return self._loss

  @property
  def bow(self):
    return self._bow

  @property
  def target_seq(self):
    return self._target_seq

  @property
  def target_seq_len(self):
    return self._target_seq_len

def save_model(path,model,reader=None):
  if reader is not None:
    with open(path+".pkl","wb+") as f:
      pickle.dump(reader,f)
  with model.graph.as_default():
    saver = tf.train.Saver()
  ckpt = os.path.basename(path+".ckpt")
  saver.save(model.session, path+".tf", latest_filename=ckpt)
  
def load_model(path):
  with open(path+".pkl","rb+") as f:
    r = pickle.load(f)
  # m = Seq2Seq(r.vocab_size,r.max_seq_len,HIDDEN_SIZE)
  m = BOW2Seq(r.vocab_size,r.max_seq_len,HIDDEN_SIZE,False)
  with m.graph.as_default():
    saver = tf.train.Saver()
  ckpt = os.path.basename(path+".ckpt")
  saver.restore(m.session, path+".tf")
  return m, r
  # ckpt = os.path.basename(path+".ckpt")
  
#################################################
#################################################
#################################################
### <CSE490>
# Play around with this if you want.
#  * EPOCHS: number of passes through the training data.
#  * Mini batch size: number of tweets to use for one training iteration.
#  * One hot vocab size: size of the vocabulary, only relevant when using one-hot representation.
#  * Hidden size: dimension of the hidden layer (assuming you implemented it :D )

EPOCHS = 25
MINI_BATCH_SIZE = 100
ONE_HOT_VOCAB_SIZE = 1500
HIDDEN_SIZE = 200

### </CSE490>

def argmax_softmax_to_str(sm,r):
  ids = sm.argmax(axis=2)
  s = r.ids_to_str(ids.tolist())
  return s
  
def test(path):
  print("Testing")
  m, r = load_model(path)
  
  total_preds = []
  total_targets = []

  for i,(x,y,y_len) in enumerate(r.yield_one_hot_batches(MINI_BATCH_SIZE,"testing")):

    cost, sm,weights = m.session.run(
      [m.cost,m.softmaxes,m.weights],
      {m.bow: x,
       m.target_seq: y,
       m.target_seq_len: y_len
      }
    )
    # embed(); exit()
    s = argmax_softmax_to_str(sm,r)
    total_preds.append(s)
    total_targets.append(r.ids_to_str(y.tolist()))
    
  preds = [p for c in total_preds for p in c]
  targets = [t.replace("<BOM> ","") for c in total_targets for t in c]

  errors = [(x,y) for x,y in zip(preds,targets) if x!=y]
  np.random.shuffle(errors)
  print()
  for pred,targ in errors[:10]:
    print("Pred:   {:s}\nTarget: {:s}\n#######".format(pred,targ))
    
  print()  
  print(len(errors),"errors;", len(preds)-len(errors),"correct")
  # embed()
  
def train(path):

  r = DataReader("stories.train.json.gz","stories.test.json.gz",vocab_size=ONE_HOT_VOCAB_SIZE)
  r.make_one_hot_vocab()

  m = BOW2Seq(r.vocab_size,r.max_seq_len,HIDDEN_SIZE)

  print("Vocab_size",r.vocab_size)
  for step in range(EPOCHS):
    print("    Starting epoch {}".format(step),file=sys.stderr)

    total_cost = 0
    for i,(x,y,y_len) in enumerate(r.yield_one_hot_batches(MINI_BATCH_SIZE)):
      cost, op, sm, weights = m.session.run(
        [m.cost,m.train_op,m.softmaxes,m.weights],
        {m.bow: x,
         m.target_seq: y,
         m.target_seq_len: y_len
        }
      )
      total_cost += cost
      if i % 10 == 0 and i != 0:
        print("Step {}, cost {}".format(i,cost),file=sys.stderr)
        # save_model(path,m,r if i == 10 else None)
        # embed();exit()
    if i % 10 != 0:
      print("Step {}, cost {}".format(i,cost),file=sys.stderr)
    save_model(path,m,r)
            
def main(path,testing=None):
  if path is None:
    raise ValueError("Provide an argument")
  elif testing is None:
    train(path)
  else:
    test(path)
if __name__ == '__main__':
  main(*sys.argv[1:])
