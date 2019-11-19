import codecs
import scipy.stats as stats
import numpy as np
import pickle
import argparse
from util import load_specialized_embeddings

def boolean_string(s):
  if s not in {'False', 'True', 'false', 'true'}:
    raise ValueError('Not a valid boolean string')
  return s == 'True' or s == 'true'


def mat_normalize(mat, norm_order=2, axis=1):
  return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])


def cosine(a, b):
  norm_a = mat_normalize(a)
  norm_b = mat_normalize(b)
  cos = np.dot(norm_a, np.transpose(norm_b))
  return cos


def load_benchmark(path="/work/anlausch/SimLex-999/SimLex-999.txt"):
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


def run_evaluation(vocab_path, vector_path, output_path, type="simlex", specialized=False):
  """
  :param vocab_path:
  :param vector_path:
  :return:
   run_evaluation("/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab", "/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vectors")
   >>> run_evaluation("/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab", "/work/anlausch/debbie/output_vectors0.0.vec")
  """
  with codecs.open(output_path, "w", "utf8") as f:
    if type == "simlex":
      f.write("SimLex-999\n")
      data = load_benchmark("/work/anlausch/SimLex-999/SimLex-999.txt")
    elif type == "simlex-gn-full":
      f.write("SimLex-GN-full\n")
      data = load_benchmark("/work/anlausch/SimLex-999/SimLex-GN-full.txt")
    elif type == "simverb":
      f.write("SimVerb-3500\n")
      data = load_benchmark("/home/anlausch/SimVerb-3500/SimVerb-3500.txt")
    if not specialized:
      vocab = load_vocab_goran(vocab_path)
    else:
      ed, vocab_list, vectors, vocab = load_specialized_embeddings(vocab_path)
    vectors = load_vectors_goran(vector_path)
    cos = cosine(vectors, vectors)

    ys = []
    scores = []
    for w1, w2, sim in data:
      if w1 in vocab and w2 in vocab:
        scores.append(cos[vocab[w1]][vocab[w2]])
        ys.append(sim)
      else:
        f.write("not in vocab %s \n" % str((w1, w2)))

    f.write("Spearman %s\n" % str(stats.spearmanr(np.array(scores), np.array(ys)).correlation))
    f.write("Pearson %s\n" % str(stats.pearsonr(np.array(scores), np.array(ys))[0]))


def main():
  parser = argparse.ArgumentParser(description="Running Simlex + Simverb")
  parser.add_argument("--output_path", type=str, default=None,
                      help="Output path", required=True)
  parser.add_argument("--embedding_vector_path", type=str, default=None,
                      help="Embedding vector path", required=True)
  parser.add_argument("--embedding_vocab_path", type=str, default=None,
                      help="Embedding vocab path", required=True)
  parser.add_argument("--specialized_embeddings", type=boolean_string, default="False",
                      help="Whether the embeddings are specialized (affects how to load them)", required=False)

  args = parser.parse_args()
  output_path_simlex = args.output_path + "/simlex.txt"
  output_path_simverb = args.output_path + "/simverb.txt"
  output_path_simlex_gn_full = args.output_path + "/simlex_gn_full.txt"
  #run_evaluation(args.embedding_vocab_path, args.embedding_vector_path, output_path=output_path_simlex_gn_full, type="simlex-gn-full")
  run_evaluation(args.embedding_vocab_path, args.embedding_vector_path, output_path=output_path_simlex, type="simlex", specialized=args.specialized_embeddings)
  run_evaluation(args.embedding_vocab_path, args.embedding_vector_path, output_path=output_path_simverb, type="simverb", specialized=args.specialized_embeddings)



if __name__=="__main__":
  main()