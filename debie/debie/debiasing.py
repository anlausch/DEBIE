import utils
import numpy as np
import cca

def get_bias_direction(equality_sets, vecs, vecs_norm, vocab, vocab_inv):
  dir_vecs = []
  for eq in equality_sets:
    if eq[0] in vocab and eq[1] in vocab:
      dir_vecs.append(vecs_norm[vocab[eq[0]]] - vecs_norm[vocab[eq[1]]])
  q = np.array(dir_vecs)
  u, sigma, v = np.linalg.svd(q)
  
  v_b = v[0]
  return v_b
 
  #scores = np.dot(v_b, np.transpose(vecs_norm))
  #order = np.argsort(scores)
  #sorted_pairs = [(vocab_inv[i], scores[i]) for i in order]
  #return sorted_pairs

def get_pis(v_b, vecs_norm):
  dots = np.dot(v_b, np.transpose(vecs_norm))
  dots = np.reshape(dots, (len(vecs_norm), 1))
  v_b_tiled = np.tile(v_b, (len(vecs_norm), 1))
  pis = np.multiply(dots, v_b_tiled)
  return pis
  
  
def debias_direction_sub(v_b, vecs_norm):
  return vecs_norm - v_b


def debias_direction_linear(v_b, vecs_norm):
  pis = get_pis(v_b, vecs_norm)
  return vecs_norm - pis

def debias_cca(equality_sets, vecs, vocab):
  A = []
  B = []
  for eq in equality_sets:
    if eq[0] in vocab and eq[1] in vocab:
      A.append(vecs[vocab[eq[0]]])
      B.append(vecs[vocab[eq[1]]])
    else:
      print(eq[0] + " and/or " + eq[1] + " not in vocab!")
  A = np.array(A); B = np.array(B);

  corr_an = cca.CCA(A, B, min(A.shape[1], B.shape[1]))
  corr_an.correlate(sklearn = True)
  proj_src, proj_trg = corr_an.transform(vecs, vecs)
  return proj_src, proj_trg  

def debias_proc(equality_sets, vecs, vocab):
  A = []
  B = []
  for eq in equality_sets:
    if eq[0] in vocab and eq[1] in vocab:
      A.append(vecs[vocab[eq[0]]])
      B.append(vecs[vocab[eq[1]]])
  A = np.array(A); B = np.array(B);

  product = np.matmul(A.transpose(), B)
  U, s, V = np.linalg.svd(product)
  print(U.shape, V.shape)
  proj_mat = V #np.matmul(U, V)
  res = np.matmul(vecs, proj_mat)
  return (res + vecs) / 2, proj_mat 