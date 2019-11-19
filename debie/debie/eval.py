import numpy as np
import utils
from scipy import stats
from sklearn.cluster import KMeans
from sklearn import svm
import random

def eval_simlex(simlex, vocab, vecs):
  preds = []
  golds = []
  cnt = 0
  for s in simlex:
    if s[0] in vocab and s[1] in vocab:
      vec1 = vecs[vocab[s[0]]]
      vec2 = vecs[vocab[s[1]]]
      sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
      preds.append(sim)
      golds.append(s[2])
    else:
      cnt += 1
      #print(s[0] + " or " + s[1] + " not in vocab")
  print("Didn't find " + str(cnt) + " pairs")
  pearson = stats.pearsonr(golds, preds)[0]
  spearman = stats.spearmanr(golds, preds)[0]
  return pearson, spearman

def eval_k_means(male_list, female_list, vecs, vocab):
  lista = male_list + female_list
  word_vecs = []
  for l in lista:
    if l in vocab:
      word_vecs.append(vecs[vocab[l]])
    else:
      print(l + " not in vocab!")
  vecs_to_cluster = word_vecs
  golds1 = [1]*len(male_list) + [0]*len(female_list)
  golds2 = [0]*len(male_list) + [1]*len(female_list)
  items = list(zip(vecs_to_cluster, golds1, golds2))

  scores = []
  for i in range(50):
    random.shuffle(items)
    kmeans = KMeans(n_clusters=2, random_state=0, init = 'k-means++').fit(np.array([x[0] for x in items]))
    preds = kmeans.labels_

    acc1 = len([i for i in range(len(preds)) if preds[i] == items[i][1]]) / len(preds)
    acc2 = len([i for i in range(len(preds)) if preds[i] == items[i][2]]) / len(preds)
    scores.append(max(acc1, acc2))
  return sum(scores) / len(scores)

def eval_svm(train_first, train_second, test_first, test_second, vecs_train, vecs_test, vocab_train, vocab_test):
  #print("SVM")
  #print(train_first, train_second)
  #print(test_first, test_second)
  #print(list(vocab_train)[:100])
  #print(list(vocab_test)[:100])  
  
  # training data preparation
  X_train = []
  y_train = []

  train = [(w, 0) for w in train_first if w in vocab_train] + [(w, 1) for w in train_second if w in vocab_train]
  print("SVM train words: " + str(len(train)))

  scores = []
  for i in range(20):
    print("Iterataion" + str(i+1))
    random.shuffle(train)
  
    for p in train:
      w = p[0]; l = p[1];
      X_train.append(vecs_train[vocab_train[w]])
      y_train.append(l)
  
    # training SVM (rbf kernel)
    clf = svm.SVC(gamma = 'scale')
    clf.fit(X_train, y_train)
  
    # prediction data preparation
    X_test = []
    y_test = []
    test = [(w, 0) for w in test_first if w in vocab_test] + [(w, 1) for w in test_second if w in vocab_test]
    #print("SVM test words: " + str(len(test)))
    for p in test:
      w = p[0]; l = p[1];
      X_test.append(vecs_test[vocab_test[w]])
      y_test.append(l)

    preds = clf.predict(X_test)
    correct = [i for i in range(len(y_test)) if y_test[i] == preds[i]]
    acc = len(correct) / len(preds)
    scores.append(acc)
  return sum(scores) / len(scores)

def embedding_coherence_test(embedding_matrix, vocab, target_1, target_2, attributes):
    sum_first = np.zeros(300)
    cnt = 0 
    for t in target_1:
      if t in vocab:
        sum_first += embedding_matrix[vocab[t]]
        cnt += 1
      else:
        print(t + " not in vocab!")
    avg_first = sum_first / float(cnt)

    sum_second = np.zeros(300)
    cnt = 0 
    for t in target_2:
      if t in vocab:
        sum_second += embedding_matrix[vocab[t]]
        cnt += 1
    avg_second = sum_second / float(cnt)

    sims_first = []
    sims_second = []
    for a in attributes:
      if a in vocab:
        vec_a = embedding_matrix[vocab[a]]
        sims_first.append(np.dot(avg_first, vec_a) / (np.linalg.norm(avg_first) * np.linalg.norm(vec_a)))
        sims_second.append(np.dot(avg_second, vec_a) / (np.linalg.norm(avg_second) * np.linalg.norm(vec_a)))
    return stats.spearmanr(sims_first, sims_second)

def bias_analogy_test(vecs, vocab, target_1, target_2, attributes_1, attributes_2):
  target_1 = [x for x in target_1 if x in vocab]
  target_2 = [x for x in target_2 if x in vocab]
  attributes_1 = [x for x in attributes_1 if x in vocab]
  attributes_2 = [x for x in attributes_2 if x in vocab]

  to_rmv = [x for x in attributes_1 if x in attributes_2]
  for x in to_rmv:
    attributes_1.remove(x)
    attributes_2.remove(x)

  if len(attributes_1) != len(attributes_2):
    #raise ValueError("Attribute lists need to be of the same length!")
    min_len = min(len(attributes_1), len(attributes_2))
    attributes_1 = attributes_1[:min_len]
    attributes_2 = attributes_2[:min_len]
  print(attributes_1)
  print(attributes_2)

  atts_paired = [] #list(zip(attributes_1, attributes_2))
  for a1 in attributes_1:
    for a2 in attributes_2:
      atts_paired.append((a1, a2))

  tmp_vocab = list(set(target_1 + target_2 + attributes_1 + attributes_2))
  dicto = []
  matrix = []
  for w in tmp_vocab:
    #print(w)
    if w in vocab:
      #print(w + " is in vocab!")
      matrix.append(vecs[vocab[w]]) 
      dicto.append(w)
  
  vecs = np.array(matrix)
  vocab = {dicto[i] : i for i in range(len(dicto))}
  
  eq_pairs = [] #list(zip(target_1, target_2))
  for t1 in target_1:
    for t2 in target_2:
      eq_pairs.append((t1, t2))
  
  for pair in eq_pairs:
    t1 = pair[0]
    t2 = pair[1]
    vec_t1 = vecs[vocab[t1]]
    vec_t2 = vecs[vocab[t2]]
      
    biased = []
    totals = []
    for a1, a2 in atts_paired:
      vec_a1 = vecs[vocab[a1]]
      vec_a2 = vecs[vocab[a2]]

      diff_vec = vec_t1 - vec_t2      

      query_1 = diff_vec + vec_a2
      query_2 = vec_a1 - diff_vec
      
      sims_q1 = np.sum(np.square(vecs - query_1), axis = 1)
      sorted_q1 = np.argsort(sims_q1)
      ind = np.where(sorted_q1 == vocab[a1])[0][0]
      other_att_2 = [x for x in attributes_2 if x != a2]
      indices_other = [np.where(sorted_q1 == vocab[x])[0][0] for x in other_att_2]
      num_bias = [x for x in indices_other if ind < x]
      biased.append(len(num_bias))
      totals.append(len(indices_other)) 

      #ranks.append(ind + 1) #ranks.append(len(vecs) - ind)

      sims_q2 = np.sum(np.square(vecs - query_2), axis = 1)
      sorted_q2 = np.argsort(sims_q2)
      ind = np.where(sorted_q2 == vocab[a2])[0][0]
      other_att_1 = [x for x in attributes_1 if x != a1]
      indices_other = [np.where(sorted_q2 == vocab[x])[0][0] for x in other_att_1]
      num_bias = [x for x in indices_other if ind < x]
      biased.append(len(num_bias))
      totals.append(len(indices_other)) 

      #ranks.append(ind + 1) #ranks.append(len(vecs) - ind)

  #return sum([1/x for x in ranks]) / len(ranks)
  return sum(biased) / sum(totals)

def bias_analogy_test_v2(vecs, vocab, target_1, target_2, attributes_1, attributes_2):
  target_1 = [x for x in target_1 if x in vocab]
  target_2 = [x for x in target_2 if x in vocab]
  attributes_1 = [x for x in attributes_1 if x in vocab]
  attributes_2 = [x for x in attributes_2 if x in vocab]

  to_rmv = [x for x in attributes_1 if x in attributes_2]
  for x in to_rmv:
    attributes_1.remove(x)
    attributes_2.remove(x)

  if len(attributes_1) != len(attributes_2):
    #raise ValueError("Attribute lists need to be of the same length!")
    min_len = min(len(attributes_1), len(attributes_2))
    attributes_1 = attributes_1[:min_len]
    attributes_2 = attributes_2[:min_len]
  print(attributes_1)
  print(attributes_2)

  atts_paired = [] #list(zip(attributes_1, attributes_2))
  for a1 in attributes_1:
    for a2 in attributes_2:
      atts_paired.append((a1, a2))

  tmp_vocab = list(set(target_1 + target_2 + attributes_1 + attributes_2))
  dicto = []
  matrix = []
  for w in tmp_vocab:
    #print(w)
    if w in vocab:
      #print(w + " is in vocab!")
      matrix.append(vecs[vocab[w]]) 
      dicto.append(w)
  
  vecs = np.array(matrix)
  vocab = {dicto[i] : i for i in range(len(dicto))}
  
  eq_pairs = [] #list(zip(target_1, target_2))
  for t1 in target_1:
    for t2 in target_2:
      eq_pairs.append((t1, t2))
  
  for pair in eq_pairs:
    t1 = pair[0]
    t2 = pair[1]
    vec_t1 = vecs[vocab[t1]]
    vec_t2 = vecs[vocab[t2]]
      
    biased = []
    totals = []
    for a1, a2 in atts_paired:
      vec_a1 = vecs[vocab[a1]]
      vec_a2 = vecs[vocab[a2]]

      diff_vec = vec_t1 - vec_a1      

      query_1 = diff_vec + vec_a2
      query_2 = vec_t2 - diff_vec
      
      sims_q1 = np.sum(np.square(vecs - query_1), axis = 1)
      sorted_q1 = np.argsort(sims_q1)
      ind = np.where(sorted_q1 == vocab[t2])[0][0]
      other_tar_1 = [x for x in target_1 if x != t1]
      indices_other = [np.where(sorted_q1 == vocab[x])[0][0] for x in other_tar_1]
      num_bias = [x for x in indices_other if ind < x]
      biased.append(len(num_bias))
      totals.append(len(indices_other)) 

      #ranks.append(ind + 1) #ranks.append(len(vecs) - ind)

      sims_q2 = np.sum(np.square(vecs - query_2), axis = 1)
      sorted_q2 = np.argsort(sims_q2)
      ind = np.where(sorted_q2 == vocab[a2])[0][0]
      other_att_1 = [x for x in attributes_1 if x != a1]
      indices_other = [np.where(sorted_q2 == vocab[x])[0][0] for x in other_att_1]
      num_bias = [x for x in indices_other if ind < x]
      biased.append(len(num_bias))
      totals.append(len(indices_other)) 

      #ranks.append(ind + 1) #ranks.append(len(vecs) - ind)

  #return sum([1/x for x in ranks]) / len(ranks)
  return sum(biased) / sum(totals)
