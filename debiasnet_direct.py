"""
Debiasnet model with direct implicit debiasing training objective (not reported in the paper)
"""
import tensorflow as tf
from flip_gradient import flip_gradient

def get_through_mlp(input, mlplayers, activ, dropout):
  mapped = input 
  for i in range(len(mlplayers)):
    mapped = tf.layers.dense(mapped, mlplayers[i], activation = activ, name = "mlplay_" + str(i+1), reuse = tf.AUTO_REUSE)
  return tf.nn.dropout(mapped, dropout)

def cosine_distance(t1, t2):
  return tf.constant(1.0, dtype = tf.float32) - tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(t1, axis = 1), tf.nn.l2_normalize(t2, axis = 1)), axis = 1)

def cosine_similarity(t1, t2):
  return tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(t1, axis = 1), tf.nn.l2_normalize(t2, axis = 1)), axis = 1)

def asym_distance(t1, t2):
    n1 = tf.norm(t1, axis = 1)
    n2 = tf.norm(t2, axis = 1)
    return tf.div(tf.subtract(n1, n2), tf.add(n1, n2))

def le_distance(t1, t2, asym_fact):
  return cosine_distance(t1, t2) + asym_fact * asym_distance(t1, t2)

def hinge_loss(true_ledists, false_ledists, margin):
  return tf.reduce_sum(tf.maximum(tf.subtract(tf.constant(margin, dtype = tf.float32), tf.subtract(false_ledists, true_ledists)), 0.0))

class DebiasNetModel(object): # setting the adversarial grad scale to -1 turns of the flipping of the gradient
  def __init__(self, embs, mlp_layers, activation = tf.nn.tanh, scope = "debie", reg_factor = 0.1, learning_rate = 0.0001, adversarial_grad_scale=1.0, i_factor=1.0, e_factor=1.0):
    self.embeddings = embs
    self.scope = scope

    with tf.name_scope(self.scope + "__placeholders"):
      # init
      self.target_1 = tf.placeholder(tf.int32, [None,], name="t1")
      self.target_2 = tf.placeholder(tf.int32, [None,], name="t2")
      self.attribute = tf.placeholder(tf.int32, [None,], name="a")

      self.dropout = tf.placeholder(tf.float32, name="dropout")

    with tf.name_scope(self.scope + "__model"):
      # embedding lookup
      self.embeddings = tf.get_variable("word_embeddings", initializer=embs, dtype = tf.float32, trainable = False)
      self.embs_target_1 = tf.nn.embedding_lookup(self.embeddings, self.target_1)
      self.embs_target_2 = tf.nn.embedding_lookup(self.embeddings, self.target_2)
      self.embs_attribute = tf.nn.embedding_lookup(self.embeddings, self.attribute)

      # MLPs (with or without shared parameters, depending on same_mapper)
      print("Mapping through MLP...")
      self.mapped_target_1 = get_through_mlp(self.embs_target_1, mlp_layers, activation, self.dropout)
      self.mapped_target_2 = get_through_mlp(self.embs_target_2, mlp_layers, activation, self.dropout)
      self.mapped_attribute = get_through_mlp(self.embs_attribute, mlp_layers, activation, self.dropout)

      print("Compute losses..")
      # Explicit debiasing objective
      self.l_e = tf.reduce_sum(tf.math.squared_difference(cosine_similarity(self.mapped_target_1, self.mapped_attribute),
                                                          cosine_similarity(self.mapped_target_2, self.mapped_attribute)))

      # regularization objective
      self.l_r = tf.reduce_sum(cosine_distance(self.embs_target_1, self.mapped_target_1)) \
                 + tf.reduce_sum(cosine_distance(self.embs_target_2, self.mapped_target_2)) \
                 + tf.reduce_sum(cosine_distance(self.embs_attribute, self.mapped_attribute))

      # new implicit debiasing objective
      self.l_i = tf.reduce_sum(cosine_distance(self.mapped_target_1, self.mapped_target_2))

      self.l_total = e_factor * self.l_e + i_factor * self.l_i + reg_factor * self.l_r
      
      self.train_step= tf.train.AdamOptimizer(learning_rate).minimize(self.l_total)
  
  def replace_embs(self, embs, session):
    assign_op = self.embeddings.assign(embs)
    session.run(assign_op)