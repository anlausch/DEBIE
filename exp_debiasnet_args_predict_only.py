import numpy as np
import tensorflow as tf
import pickle
import os
import argparse
import model_old as model

def boolean_string(s):
  if s not in {'False', 'True', 'false', 'true'}:
    raise ValueError('Not a valid boolean string')
  return s == 'True' or s == 'true'

def create_dir_if_not_exists(path):
  if not os.path.exists(path):
      os.makedirs(path)

parser = argparse.ArgumentParser(description="Running DEBIE prediction")
parser.add_argument("--output_path", type=str, default=None,
                    help="Output path for the vectors", required=True)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Checkpoint", required=True)
parser.add_argument("--embedding_vector_path", type=str, default=None,
                    help="Embedding vector path", required=True)
parser.add_argument("--lang", type=str, default=False,
                    help="language", required=False)

args = parser.parse_args()



output_path = args.output_path
embedding_vector_path = args.embedding_vector_path




create_dir_if_not_exists(output_path)
PARAMETERS = { "model_name": "exp_deb_" + args.checkpoint + ".model",
               "ckpt_path" : output_path,
               "emb_size": 300,
               "mlp_lay" : [300]*5,
               "dropout": 1.0,
               "reg_factor": 1.0,
               "learning_rate": 0.0001,
               "num_dev": 2000,
               "batch_size": 50,
               "eval_steps": 1000,
               "num_evals_not_better_end": 10,
               "e_f": 1.0,
               "i_f": 1.0}



print("Loading embeddings...")
vectors = np.load(embedding_vector_path, allow_pickle=True).astype(np.float32)


while len(vectors) < 199984:
  vectors = np.append(vectors, np.array([np.zeros(300, dtype=np.float32)], dtype=np.float32), axis=0)
vectors = vectors[:199984]
reshaped_vectors = vectors
### TODO: Eventuell muss ich alle Werte noch einmal berechnen; nehme ich nicht den besten chkpint? --> alle vektoren einmal mappen


class modelExecutor:
  def __init__(self):
    tf.reset_default_graph()
    # model initialization
    self.model = model.DebbieModel(vectors, PARAMETERS["mlp_lay"], activation = tf.nn.tanh, scope = "debbie", learning_rate = PARAMETERS["learning_rate"], reg_factor=PARAMETERS["reg_factor"], adversarial=False)#, e_factor=PARAMETERS["e_f"], i_factor=PARAMETERS["i_f"])

    self.init = tf.global_variables_initializer()
    self.sess = None
    self.saver = tf.train.Saver()


  def output_vectors(self):
    self.sess = tf.Session()
    self.sess.run(self.init)

    # Restore most recent checkpoint if it exists.
    ckpt_file = os.path.join(PARAMETERS["ckpt_path"], PARAMETERS["model_name"]) + ".ckpt"
    if os.path.isfile(ckpt_file + ".meta"):
      if os.path.isfile(ckpt_file + "_best.meta"):
        self.saver.restore(self.sess, (ckpt_file + "_best"))
    self.model.replace_embs(vectors, self.sess)
    self.retrieve_vectors()



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
    pickle.dump(new_vecs, open(output_path + "/" + str(args.lang) + "_debiased.vectors", "wb"))
    return new_vecs


me = modelExecutor()
me.output_vectors()

