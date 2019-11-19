import numpy as np

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