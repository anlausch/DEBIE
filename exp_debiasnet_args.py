import numpy as np
import tensorflow as tf
import pickle
import sys
import random
import logger
import os
import itertools
from util import load_specialized_embeddings
import argparse
import data_handler

def boolean_string(s):
  if s not in {'False', 'True', 'false', 'true'}:
    raise ValueError('Not a valid boolean string')
  return s == 'True' or s == 'true'

def create_dir_if_not_exists(path):
  if not os.path.exists(path):
      os.makedirs(path)

parser = argparse.ArgumentParser(description="Running DEBIE")
parser.add_argument("--dropout_keep_probs", type=str, help="List of dropout keep probability values", required=True)
parser.add_argument("--reg_factors", type=str, default=None,
                    help="List of regularization factors", required=True)
parser.add_argument("--output_path", type=str, default=None,
                    help="Output path", required=True)
parser.add_argument("--input_path", type=str, default=None,
                    help="Input path in case train and dev are in a single file", required=False)
parser.add_argument("--input_path_train", type=str, default=None,
                    help="Input path for a dedicated train set", required=False)
parser.add_argument("--input_path_dev", type=str, default=None,
                    help="Input path for a dedicated dev set", required=False)
parser.add_argument("--embedding_vector_path", type=str, default=None,
                    help="Embedding vector path", required=True)
parser.add_argument("--embedding_vocab_path", type=str, default=None,
                    help="Embedding vocab path", required=True)
parser.add_argument("--e_factors", type=str, default=None,
                    help="List of weights for the explicit debiasing objective", required=True)
parser.add_argument("--i_factors", type=str, default=None,
                    help="List of weights for the implicit debiasing objective", required=True)
parser.add_argument("--specialized_embeddings", type=boolean_string, default=False,
                    help="Whether the input embeddings are specialized (requires a different way of loading them)", required=False)
parser.add_argument("--direct_implicit_objective", type=boolean_string, default=False,
                    help="Whether to use the adversarial (False) or the direct implicit debiasing objective (True)", required=False)

args = parser.parse_args()


drop_vals = eval(args.dropout_keep_probs)
reg_factors = eval(args.reg_factors)
i_factors = eval(args.i_factors)
e_factors = eval(args.e_factors)
output_path = args.output_path
if args.input_path is not None:
  input_path = args.input_path
  data_mode = "single"
elif args.input_path_train is not None and args.input_path_dev is not None:
  input_path_train = args.input_path_train
  input_path_dev = args.input_path_dev
  data_mode = "splitted"
else:
  raise ValueError("Either a single file or a train and a dev file need to be supplied")
embedding_vector_path = args.embedding_vector_path
embedding_vocab_path = args.embedding_vocab_path
specialized_embeddings = args.specialized_embeddings
direct_implicit_objective = args.direct_implicit_objective
if direct_implicit_objective:
  import debiasnet_direct as model
else:
  import debiasnet as model

random.seed(1000)

create_dir_if_not_exists(output_path)
configs = list(itertools.product(drop_vals, reg_factors, e_factors, i_factors))
print(configs[0])
print(len(configs))

for drp, rf, e_f, i_f in configs:
  print("Configuration: ")
  print(drp, rf, e_f, i_f)
  print()
  print()
  config_string = "drp=" + str(drp) + "_rf=" + str(rf) + "_ef=" + str(e_f) + "_if=" + str(i_f)
  special_output_path = output_path + "/" + config_string
  create_dir_if_not_exists(special_output_path)
  # TODO: Check if output path exists if not create it
  PARAMETERS = { "model_name": "exp_deb_" + config_string + ".model",
                 "log_path": special_output_path + "/" + config_string + ".log",
                 "ckpt_path" : special_output_path,
                 "emb_size": 300,
                 "mlp_lay" : [300]*5,
                 "dropout": drp,
                 "reg_factor": rf,
                 "learning_rate": 0.0001,
                 "num_dev": 2000,
                 "batch_size": 50,
                 "eval_steps": 1000,
                 "num_evals_not_better_end": 10,
                 "e_f": e_f,
                 "i_f": i_f}

  print("Loading data...")
  if data_mode == "single":
    print("Running single file data mode")
    data = data_handler.load_input_examples(input_path)
    random.shuffle(data)
    split_index = int(len(data)*0.7)
    train = data[:split_index]
    dev = data[split_index:]
  elif data_mode == "splitted":
    print("Running splitted data mode")
    train = data_handler.load_input_examples(input_path_train)
    dev = data_handler.load_input_examples(input_path_dev)

  print("Loading embeddings...")
  if not specialized_embeddings:
    vectors = np.load(embedding_vector_path, allow_pickle=True).astype(np.float32)
    vocab = pickle.load(open(embedding_vocab_path,"rb"))
  else:
    ed, vocab_list, vectors, vocab = load_specialized_embeddings(embedding_vector_path)

  reshaped_vectors = vectors

  logg = logger.Logger(PARAMETERS["log_path"])


  class modelExecutor:
    def __init__(self):
      tf.reset_default_graph()
      # model initialization
      self.model = model.DebiasNetModel(vectors, PARAMETERS["mlp_lay"], activation = tf.nn.tanh, scope ="debbie", learning_rate = PARAMETERS["learning_rate"], reg_factor=PARAMETERS["reg_factor"], e_factor=PARAMETERS["e_f"], i_factor=PARAMETERS["i_f"])
      self.batch_size = PARAMETERS["batch_size"]
      self.keep_rate = PARAMETERS["dropout"]
      self.eval_steps = PARAMETERS["eval_steps"]

      logg.Log("Initializing variables")
      self.init = tf.global_variables_initializer()
      self.sess = None
      self.saver = tf.train.Saver()

    def get_minibatch(self, triples):
        t1s = []; t2s = []; aas = []
        for t in triples:
          # This is only relevant for the postspec embedding space as I want to use exactly the same data, but I filtered the vocab with the original fasttext top 200k
          if t.t1 in vocab and t.t2 in vocab and t.a in vocab:
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
          logg.Log("Restored best dev loss: %f" % (self.best_dev))
          self.saver.restore(self.sess, ckpt_file)
          logg.Log("Model restored from file: %s" % ckpt_file)

      ### Training cycle
      logg.Log("Training...")
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


          cl, cle, clr, cli = self.sess.run([self.model.l_total, self.model.l_e, self.model.l_r, self.model.l_i], feed_dict)
          logg.Log("Total loss: %s, Explicit loss: %s, Implicit loss: %s, Regularization loss: %s" % (str(cl), str(cle), str(cli), str(clr)))

          _, c = self.sess.run([self.model.train_step, self.model.l_total], feed_dict)

          epoch_loss += c

          if self.step % self.eval_steps == 0:
            dev_perf = self.eval()
            logg.Log("Step: %i\t Dev perf: %f" % (self.step, dev_perf))
            self.saver.save(self.sess, ckpt_file)

            print("Saving model...")
            if dev_perf < 0.999 * self.best_dev:
              print("New best model found...")
              print("New best dev loss: " + str(dev_perf))
              self.saver.save(self.sess, ckpt_file + "_best")
              self.best_dev = dev_perf
              self.best_step = self.step
              logg.Log("Checkpointing with new best matched-dev loss: %f" % (self.best_dev))
            elif (self.step > self.best_step + (self.eval_steps * PARAMETERS["num_evals_not_better_end"] + 10)):
              print("Exit condition (early stopping) met.")
              logg.Log("Best matched-dev loss: %s" % (self.best_dev))
              return

            #if self.step % 4000 == 0 and self.step != 0:
            #  reshaped_vectors = self.update_embs_neg_examples()

          self.step += 1

        # Display some statistics about the epoch
        logg.Log("Epoch: %i\t Avg. Cost: %f" % (self.epoch + 1, epoch_loss / len(train)))
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
  pickle.dump(new_vectors, open(special_output_path + "/" + str(config_string) + ".vec", "wb"))

