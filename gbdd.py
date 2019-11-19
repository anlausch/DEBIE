import numpy as np


def get_bias_direction(equality_sets, vecs, vecs_norm, vocab, vocab_inv):
  dir_vecs = []
  for eq in equality_sets:
    if eq[0] in vocab and eq[1] in vocab:
      dir_vecs.append(vecs_norm[vocab[eq[0]]] - vecs_norm[vocab[eq[1]]])
  q = np.array(dir_vecs)
  u, sigma, v = np.linalg.svd(q)

  v_b = v[0]
  return v_b

  # scores = np.dot(v_b, np.transpose(vecs_norm))
  # order = np.argsort(scores)
  # sorted_pairs = [(vocab_inv[i], scores[i]) for i in order]
  # return sorted_pairs


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

