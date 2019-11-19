import numpy as np
import random
import logging
import math
from itertools import filterfalse
from itertools import combinations
import argparse
import time
import pickle
import codecs
import os
import utils

# TODO: This is highly experimental
def build_vocab_dict(test_vocab, embedding_dict):
  vocab = {}
  test_vocab_set = set(test_vocab)
  index = 0
  for term in test_vocab_set:
    if term in embedding_dict:
      vocab[term] = index
      index += 1
    else:
      logging.warning("Not in vocab %s", term)
  return vocab


def convert_by_vocab(items, vocab):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    if item in vocab:
      output.append(vocab[item])
    else:
      continue
  return output


def mat_normalize(mat, norm_order=2, axis=1):
  return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])


def cosine(a, b):
  norm_a = mat_normalize(a)
  norm_b = mat_normalize(b)
  cos = np.dot(norm_a, np.transpose(norm_b))
  return cos


def build_embedding_matrix(vocab, embd_dict):
  embedding_matrix = []
  for term, index in vocab.items():
    if term in embd_dict:
      embedding_matrix.append(embd_dict[term])
    else:
      raise AssertionError("This should not happen.")
  return embedding_matrix


def similarity_lookup(w1, w2, similarities):
  return similarities[w1, w2]

def init_similarities(embedding_matrix):
    return cosine(embedding_matrix, embedding_matrix)


def normalized_association_score(w, A, B, similarities):
  return (np.mean([similarity_lookup(w, a, similarities) for a in A]) - np.mean([similarity_lookup(w, b, similarities) for b in B]))/ \
         np.std([similarity_lookup(w, x, similarities) for x in A + B])


def test_statistic(W, A, B, similarities):
  return np.mean([normalized_association_score(w, A, B, similarities) for w in W])


def random_permutation(iterable, r=None):
  pool = tuple(iterable)
  r = len(pool) if r is None else r
  return tuple(random.sample(pool, r))


def p_value(W, A, B, sample, similarities):
  logging.info("Calculating p value ... ")
  size_of_permutation = min(len(A), len(B))
  A_B = A + B
  observed_test_stats_over_permutations = []
  total_possible_permutations = math.factorial(len(A_B)) / math.factorial(size_of_permutation) / math.factorial((len(A_B)-size_of_permutation))
  logging.info("Number of possible permutations: %d", total_possible_permutations)
  if not sample or sample >= total_possible_permutations:
    permutations = combinations(A_B, size_of_permutation)
  else:
    logging.info("Computing randomly first %d permutations", sample)
    permutations = set()
    while len(permutations) < sample:
      permutations.add(tuple(sorted(random_permutation(A_B, size_of_permutation))))

  for Ai in permutations:
    Bi = filterfalse(lambda w: w in Ai, A_B)
    observed_test_stats_over_permutations.append(test_statistic(W, list(Ai), list(Bi), similarities))
    if len(observed_test_stats_over_permutations) % 100000 == 0:
      logging.info("Iteration %s finished", str(len(observed_test_stats_over_permutations)))
  unperturbed = test_statistic(W, A, B, similarities)
  is_over = np.array([o > unperturbed for o in observed_test_stats_over_permutations])
  return is_over.sum() / is_over.size


def st_weat_stats(W, A, B, similarities, sample_p=None):
  # TODO: What is the effect size in our case?
  t = test_statistic(W, A, B, similarities)
  p = p_value(W, A, B, sample=sample_p, similarities=similarities)
  return t, p


def run_test(targets, attributes_1, attributes_2, embd_dict, sample_p=None):
  """Run the WEAT test for differential association between two
  sets of target words and two sets of attributes.

  RETURNS:
      (d, e, p). A tuple of floats, where d is the WEAT Test statistic,
      e is the effect size, and p is the one-sided p-value measuring the
      (un)likeliness of the null hypothesis (which is that there is no
      difference in association between the two target word sets and
      the attributes).

      If e is large and p small, then differences in the model between
      the attribute word sets match differences between the targets.
  """
  test_vocab = targets + attributes_1 + attributes_2
  vocab_dict = build_vocab_dict(test_vocab, embd_dict)
  T = convert_by_vocab(targets, vocab_dict)
  A = convert_by_vocab(attributes_1, vocab_dict)
  B = convert_by_vocab(attributes_2, vocab_dict)
  while len(A) < len(B):
    logging.info("Popped A2 %d", B[-1])
    B.pop(-1)
  while len(B) < len(A):
    logging.info("Popped A1 %d", A[-1])
    A.pop(-1)
  assert len(A) == len(B)
  embedding_matrix = build_embedding_matrix(vocab_dict, embd_dict)
  similarities = init_similarities(embedding_matrix)
  return st_weat_stats(T, A, B, similarities, sample_p)


def test_1():
  # ggay destroys the test
  # with the two targeted test, you might be able to see historic bias such as homosexual vs queer
  targets = ["homosexual", "bisexual", "asexual", "queer", "homo", "LGBT"]
  #attributes_1 = ["joy", "love", "peace", "wonderful", "pleasure", "friend", "laughter", "happy", "natural", "normal"] "gay", "dyke", "butch", "bear", "fairy"
  #attributes_2 = ["agony", "terrible", "horrible", "nasty", "evil", "war", "awful", "failure", "unnatural", "abnormal"]
  #attributes_2 = ["caress", "freedom", "health", "peace", "cheer", "friend", "heaven", "loyal", "pleasure",
  #                "diamond", "gentle", "honest", "lucky", "rainbow", "diploma", "gift", "honor", "miracle", "sunrise",
  #                "family", "happy", "laughter", "paradise", "vacation", "child"]
  #attributes_1 = ["abuse", "crash", "filth", "murder", "illness", "accident", "death", "grief", "poison", "stink",
  #                "assault", "disaster", "pollute", "tragedy", "jail", "poverty", "ugly",
  #                "cancer","kill", "rotten","vomit", "agony", "prison", "hell","evil"]
  # check whhether this is really fair!!!!

  # significant: changed sickness to illness (0.3724263, 0.04608)
  # removed rainbow as it is a symbol (0.38907862, 0.04636)
  # added gay lesbian (0.3336907, 0.08379)
  # removed gay lesbian
  # maybe remove asexual
  attributes_2 = ["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal", "pleasure",
                  "diamond", "gentle", "honest", "lucky", "diploma", "gift", "honor", "miracle", "sunrise",
                  "family", "happy", "laughter", "paradise", "vacation"]
  # desease?
  attributes_1 = ["abuse", "crash", "filth", "murder", "illness", "accident", "death", "grief", "poison", "stink",
                  "assault", "disaster", "hatred", "pollute", "tragedy", "divorce", "jail", "poverty", "ugly", "cancer",
                  "kill", "rotten",
                  "vomit", "agony", "prison"]
  #attributes_1 = ["agony", "awful", "horrible", "humiliate", "nasty", "painful", "terrible", "tragic", "ill", "evil", "prison", "hell"]
  #attributes_2 = ["beautiful", "glorious", "joyful", "lovely", "marvelous", "pleasure","wonderful", "healthy", "peaceful", "freedom", "heaven"]
  return targets, attributes_1, attributes_2


def test_2():
  targets = ["homosexual", "bisexual", "queer", "homo", "LGBT"]
  attributes_1 = ["agony", "awful", "horrible", "humiliate", "nasty", "painful", "terrible", "tragic", "ill", "evil", "prison", "hell"]
  attributes_2 = ["beautiful", "glorious", "joyful", "lovely", "marvelous", "pleasure","wonderful", "healthy", "peaceful", "freedom", "heaven"]
  return targets, attributes_1, attributes_2


def load_vocab_goran(path):
  return pickle.load(open(path, "rb"))


def load_vectors_goran(path):
  return np.load(path)


def load_embedding_dict(vocab_path="", vector_path="", glove=False):
  """
  >>> _load_embedding_dict()
  :param vocab_path:
  :param vector_path:
  :return: embd_dict
  """
  embd_dict = {}
  if glove:
    if os.name == "nt":
      embd_dict = utils.load_embeddings("C:/Users/anlausch/workspace/embedding_files/glove.6B/glove.6B.300d.txt",
                                        word2vec=False)
    else:
      embd_dict = utils.load_embeddings("/work/anlausch/glove.6B.300d.txt", word2vec=False)
  else:
    vocab = load_vocab_goran(vocab_path)
    vectors = load_vectors_goran(vector_path)
    for term, index in vocab.items():
      embd_dict[term] = vectors[index]
    assert len(embd_dict) == len(vocab)
  return embd_dict


def main():
  def boolean_string(s):
    if s not in {'False', 'True', 'false', 'true'}:
      raise ValueError('Not a valid boolean string')
    return s == 'True' or s == 'true'
  parser = argparse.ArgumentParser(description="Running ST_WEAT")
  parser.add_argument("--test_number", type=int, help="Number of the st weat test to run (currently only 1 test is supported)", required=False)
  parser.add_argument("--permutation_number", type=int, default=None,
                      help="Number of permutations (otherwise all will be run)", required=False)
  parser.add_argument("--output_file", type=str, default=None, help="File to store the results)", required=False)
  parser.add_argument("--lower", type=boolean_string, default=False, help="Whether to lower the vocab", required=True)
  parser.add_argument("--embedding_vocab", type=str, help="Vocab of the embeddings")
  parser.add_argument("--embedding_vectors", type=str, help="Vectors of the embeddings")
  parser.add_argument("--use_glove", type=boolean_string, default=False, help="Use glove")
  args = parser.parse_args()

  start = time.time()
  logging.basicConfig(level=logging.INFO)
  logging.info("ST WEAT started")
  if args.test_number == 1:
    targets, attributes_1, attributes_2 = test_1()
  elif args.test_number == 2:
    targets, attributes_1, attributes_2 = test_2()
  else:
    raise ValueError("Only ST WEAT 1 is supported")

  if args.lower:
    targets = [t.lower() for t in targets]
    attributes_1 = [a.lower() for a in attributes_1]
    attributes_2 = [a.lower() for a in attributes_2]

  if args.use_glove:
    logging.info("Using glove")
    embd_dict = load_embedding_dict(glove=True)
  else:
    embd_dict = load_embedding_dict(vocab_path=args.embedding_vocab, vector_path=args.embedding_vectors)

  logging.info("Embeddings loaded")
  logging.info("Running test")
  result = run_test(targets, attributes_1, attributes_2, embd_dict, args.permutation_number)
  logging.info(result)
  with codecs.open(args.output_file, "w", "utf8") as f:
    f.write("Config: ")
    f.write(str(args.test_number) + " and ")
    f.write(str(args.lower) + " and ")
    f.write(str(args.permutation_number) + "\n")
    f.write("Result: ")
    f.write(str(result))
    f.write("\n")
    end = time.time()
    duration_in_hours = ((end - start) / 60) / 60
    f.write(str(duration_in_hours))
    f.close()

if __name__ == "__main__":
  main()