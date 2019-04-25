import util
import model
import numpy as np
import tensorflow as tf
import pickle
import sys
import random
import logger
from scipy import stats
import os
from sys import stdin
import itertools
import data_handler

drop_vals = [0.5, 1.0]
reg_factor = [0.1, 0.5, 1.0]

configs = list(itertools.product(drop_vals, reg_factor))
print(configs[0])
print(len(configs))

config_ind = int(sys.argv[1])
drp, rf = configs[config_ind]
print("Configuration: ")
print(drp, rf)
print()
print()

suffix = sys.argv[2]

PARAMETERS = { "model_name": "exp_deb_" + str(config_ind) + "." + suffix +  ".model",
               "log_path": "/work/anlausch/debbie/output_" + str(config_ind) + "." + suffix + ".log",
               "ckpt_path" : "/work/anlausch/debbie/output_",
               "emb_size": 300, 
               "mlp_lay" : [300]*5,
               "dropout": drp,
               "reg_factor": rf,
               "learning_rate": 0.0001,
               "num_dev": 2000,
               "batch_size": 50,
               "eval_steps": 1000,
               "num_evals_not_better_end": 10}       

print("Loading data...")
data = data_handler.load_input_examples("/work/anlausch/debbie/data/weat_1_prepared_filtered_small.txt")
train = data[:800]
dev = data[800:]

print("Loading embeddings...")
vectors = np.load("/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vectors")
vocab = pickle.load(open("/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab","rb")) 
reshaped_vectors = vectors

logger = logger.Logger(PARAMETERS["log_path"])


class modelExecutor:
  def __init__(self):
    # model initialization
    self.model = model.DebbieModel(vectors, PARAMETERS["mlp_lay"], activation = tf.nn.tanh, scope = "debbie", learning_rate = PARAMETERS["learning_rate"], reg_factor=PARAMETERS["reg_factor"])
    self.batch_size = PARAMETERS["batch_size"]
    self.keep_rate = PARAMETERS["dropout"]
    self.eval_steps = PARAMETERS["eval_steps"]

    logger.Log("Initializing variables")
    self.init = tf.global_variables_initializer()
    self.sess = None
    self.saver = tf.train.Saver()

  def get_minibatch(self, triples):
      t1s = []; t2s = []; aas = []
      for t in triples:
        ind_t1 = vocab[t.t1]
        ind_t2 = vocab[t.t2]
        ind_a = vocab[t.a]
        t1s.append(ind_t1)
        t2s.append(ind_t2)
        aas.append(ind_a)
      return t1s, t2s, aas

  def train_model(self):
    self.step = 0
    self.epoch = 0
    self.best_dev = 100000000
    self.best_mtrain = 100000000
    self.last_train = [1000000, 1000000, 1000000, 1000000, 1000000]
    self.best_step = 0

    self.sess = tf.Session()
    self.sess.run(self.init)

    # Restore most recent checkpoint if it exists. 
    ckpt_file = os.path.join(PARAMETERS["ckpt_path"], PARAMETERS["model_name"]) + ".ckpt"
    if os.path.isfile(ckpt_file + ".meta"):
      if os.path.isfile(ckpt_file + "_best.meta"):
        self.saver.restore(self.sess, (ckpt_file + "_best"))
        self.best_dev = self.eval()
        logger.Log("Restored best dev loss: %f" % (self.best_dev))
        self.saver.restore(self.sess, ckpt_file) 
        logger.Log("Model restored from file: %s" % ckpt_file)

    ### Training cycle
    logger.Log("Training...")
    reshaped_vectors = vectors
    while True:
      epoch_loss = 0.0

      random.shuffle(train)

      num_batch = int(len(train) / self.batch_size) if len(train) % self.batch_size == 0 else (int(len(train) / self.batch_size) + 1)
      batches = [train[i * self.batch_size : (i+1) * self.batch_size] for i in range(num_batch)]
      print(len(batches))

      random.shuffle(batches)

      # Loop over all batches in epoch
      for batch in batches:
        t1s, t2s, aas = self.get_minibatch(batch)
                
        # Run the optimizer to take a gradient step, and also fetch the value of the 
        # cost function for logging
        feed_dict = {self.model.target_1: t1s,
                     self.model.target_2: t2s,
                     self.model.attribute: aas,
                     self.model.dropout: self.keep_rate }
         

        cl, cle, clr = self.sess.run([self.model.l_total, self.model.l_e, self.model.l_r], feed_dict)
        _, c = self.sess.run([self.model.train_step, self.model.l_total], feed_dict)

        epoch_loss += c

        if self.step % self.eval_steps == 0:
          dev_perf = self.eval()
          logger.Log("Step: %i\t Dev perf: %f" %(self.step, dev_perf))
          self.saver.save(self.sess, ckpt_file)

          print("Saving model...")  
          if dev_perf < 0.999 * self.best_dev:
            print("New best model found...") 
            print("New best dev loss: " + str(dev_perf)) 
            self.saver.save(self.sess, ckpt_file + "_best")
            self.best_dev = dev_perf
            self.best_step = self.step
            logger.Log("Checkpointing with new best matched-dev loss: %f" %(self.best_dev))
          elif (self.step > self.best_step + (self.eval_steps * PARAMETERS["num_evals_not_better_end"] + 10)):
            print("Exit condition (early stopping) met.")  
            logger.Log("Best matched-dev loss: %s" % (self.best_dev))
            return

          #if self.step % 4000 == 0 and self.step != 0:
          #  reshaped_vectors = self.update_embs_neg_examples()

        self.step += 1
                                  
      # Display some statistics about the epoch
      logger.Log("Epoch: %i\t Avg. Cost: %f" %(self.epoch+1, epoch_loss / len(train)))
      self.epoch += 1 
      epoch_loss = 0.0 


  def retrieve_vectors(self):
    all_inds = list(range(len(vectors)))
    bs = 5000
    total_batch = int(len(all_inds) / bs) if len(all_inds) % bs == 0 else (int(len(all_inds) / bs) + 1)
    for i in range(total_batch):
      print("Batch" + str(i+1))
      w1s = all_inds[bs * i: bs * (i + 1)]
      feed_dict = {self.model.target_1: w1s,
                   self.model.dropout: 1.0 }
      retrieved_vecs = self.sess.run(self.model.mapped_target_1, feed_dict)
      print(len(retrieved_vecs)) 
      if i == 0: 
        new_vecs = retrieved_vecs
      else: 
        new_vecs = np.vstack([new_vecs, retrieved_vecs])    
    print("Created updated vectors for the whole vocabulary")       
    print(len(new_vecs), new_vecs.shape)  
    return new_vecs


  def eval(self):
    total_dev_loss = 0.0
    dev_num_batch = int(len(dev) / self.batch_size) if len(dev) % self.batch_size == 0 else (int(len(dev) / self.batch_size) + 1)
    dev_batches = [dev[i * self.batch_size : (i+1) * self.batch_size] for i in range(dev_num_batch)]
    

    random.shuffle(dev_batches) 

    # Loop over all batches in dev set
    total_dev_loss = 0.0
    for db in dev_batches:
      t1, t2, a = self.get_minibatch(db)
                

      feed_dict = {self.model.target_1: t1,
                   self.model.target_2: t2,
                   self.model.attribute: a,
                   self.model.dropout: self.keep_rate }

      c = 0.0
      c = self.sess.run(self.model.l_total, feed_dict)
      total_dev_loss += c
    return total_dev_loss / len(dev)

me = modelExecutor()
me.train_model()
new_vectors = me.retrieve_vectors()
pickle.dump(new_vectors, open("/work/anlausch/debbie/output_vectors" + str(config_ind) + "." + suffix + ".vec", "wb"))

