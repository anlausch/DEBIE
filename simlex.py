import codecs
import scipy.stats as stats
import numpy as np
import pickle

def mat_normalize(mat, norm_order=2, axis=1):
  return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])


def cosine(a, b):
  norm_a = mat_normalize(a)
  norm_b = mat_normalize(b)
  cos = np.dot(norm_a, np.transpose(norm_b))
  return cos


def load_simlex(path="/work/anlausch/SimLex-999/SimLex-999.txt"):
  data = []
  with codecs.open(path, "r", "utf8") as f:
    for i, line in enumerate(f.readlines()):
      # omit header
      if i != 0:
        parts = line.split("\t")
        w1 = parts[0]
        w2 = parts[1]
        sim = float(parts[3])
        data.append([w1, w2, sim])
  return data


def load_vocab_goran(path):
  return pickle.load(open(path, "rb"))


def load_vectors_goran(path):
  return np.load(path)


def run_evaluation(vocab_path, vector_path):
  """
  :param vocab_path:
  :param vector_path:
  :return:
   run_evaluation("/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab", "/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vectors")
   >>> run_evaluation("/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab", "/work/anlausch/debbie/output_vectors0.0.vec")
  """
  data = load_simlex()
  vocab = load_vocab_goran(vocab_path)
  vectors = load_vectors_goran(vector_path)
  cos = cosine(vectors, vectors)

  ys = []
  scores = []
  for w1, w2, sim in data:
    if w1 in vocab and w2 in vocab:
      scores.append(cos[vocab[w1]][vocab[w2]])
      ys.append(sim)
    else:
      print("not in vocab %s" % w1 + " " + w2)

  print("Spearman", stats.spearmanr(np.array(scores), np.array(ys)).correlation)
  print("Pearson", stats.pearsonr(np.array(scores), np.array(ys))[0])

def main():
  run_evaluation()


if __name__=="__main__":
  main()