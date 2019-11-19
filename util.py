import codecs
import os
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import numpy as np
import random
import pickle
from utils import load_embeddings

def output_specialized_embedding_vocab(ipath, opath):
  """
  >>> output_specialized_embedding_vocab("/home/anlausch/post-specialized-embeddings/postspec/ft_postspec.txt", "/work/anlausch/debbie/output/")
  """
  embedding_dict, vocab_list, vector_list, word2index = load_specialized_embeddings(ipath)
  pickle.dump(word2index, open(opath + "ft_postspec.vocab", "wb"))
  vector_list.dump(opath + "ft_postspec.vec")


def output_glove_embedding_vocab(ipath, opath):
  """
  #>>> output_glove_embedding_vocab("/work/anlausch/glove.6B.300d.txt", "/work/anlausch/debbie/output/")
  >>> output_glove_embedding_vocab("/work/gglavas/data/word_embs/yacle/cbow/cbow.wiki.en.300w5.vec", "/work/anlausch/debbie/output/")
  """
  embedding_dict = load_embeddings(ipath)
  word_2_index = {}
  vector_list = []
  i = 0
  for j,(term, vec) in enumerate(embedding_dict.items()):
    # this is w2v specific
    if j != 0:
      if i < 200000:
        if len(vec) == 300:
          word_2_index[term] = i
          vector_list.append(np.array(vec).astype(np.float32))
          i+=1
        else:
          print(i)
          print(vec)
      else:
        break
  embedding_dict = None
  vector_list = np.array(vector_list).astype(np.float32)
  pickle.dump(word_2_index, open(opath + "w2v_cbow_200k.vocab", "wb"))
  vector_list.dump(opath + "w2v_cbow_200k.vec")




### RAW IO
# returns a dictionary of embeddings
def load_specialized_embeddings(path):
    """
    :param path:
    :return:
    """
    embbedding_dict = {}
    vocab_list = []
    vector_list = []
    word2index = {}
    with codecs.open(path, "rb", "utf8", "ignore") as infile:
        for line in infile:
            try:
                parts = line.split()
                word = parts[0].split("en_")[1]
                nums = np.array([float(p) for p in parts[1:]])
                embbedding_dict[word] = nums
                vocab_list.append(word)
                vector_list.append(nums)
                word2index[word] = len(vocab_list)-1
            except Exception as e:
                print(line)
                continue
    assert("test" in embbedding_dict)
    assert ("house" in embbedding_dict)
    return embbedding_dict, np.array(vocab_list), np.array(vector_list, dtype=np.float32), word2index

"""
this very first function is not for augmentation, but for selecting random "attribute" terms from the vocabulary
I could either include the original lists or not --> if yes, then I can make sure, that they are not in the target lists already
"""
def select_random_attributes(vocab_list, k, original_weat_path):
  """
  :param vocab_list:
  :param k:
  :param original_weat_path:
  :return:
   >>> select_random_attributes()
  """
  # get only the most frequent 50k terms
  random_attributes = []
  t1, t2, a1, a2 = fuse_stimuli([original_weat_path])
  reserved_vocab = t1 + t2 + a1 + a2
  random.seed(1000)
  vocab_list = vocab_list[:50000]
  samples = random.choices(vocab_list, k=(k+len(reserved_vocab)))
  for sample in samples:
    if sample not in reserved_vocab and len(random_attributes) < k:
      random_attributes.append(sample)
  return random_attributes


def fuse_stimuli(paths):
  """
  :param paths:
  :return:
   >>> fuse_stimuli(["./data/weat_1.txt", "./data/weat_1_augmentation.txt"])
  """
  stimuli_dicts = []
  for path in paths:
    stimuli_dicts.append(read_weat_data(path))
  t1s = []
  t2s = []
  a1s = []
  a2s = []
  for stimuli_dict in stimuli_dicts:
    for key, value in stimuli_dict.items():
      if key == "T1:":
        t1s.append(value)
      elif key == "T2:":
        t2s.append(value)
      elif key == "A1:":
        a1s.append(value)
      elif key == "A2:":
        a2s.append(value)
      else:
        raise ValueError("Key does not match expected keys")
  t1 = flatten(t1s)
  t2 = flatten(t2s)
  a1 = flatten(a1s)
  a2 = flatten(a2s)
  return t1, t2, a1, a2



def read_weat_data(path):
  """
  :param path:
  :return:
   >>> read_weat_data("./data/weat_1.txt")
  """
  stimuli = {}
  with codecs.open(path, "r", "utf8") as f:
    for line in f.readlines():
      key = line.strip().split(" ")[0]
      data = line.strip().split(" ")[1:]
      stimuli[key] = data
  return stimuli


def flatten(l):
  return [item for sublist in l for item in sublist]


def get_directory_files(dir_path):
    files = [f for f in list(os.walk(dir_path))[0][2]]
    return (files, [os.path.join(dir_path, f) for f in files])

def load_lines(path):
	return [l.strip() for l in list(codecs.open(path, "r", encoding = 'utf8', errors = 'replace').readlines())]

def write_lines(path, list, append = False):
	f = codecs.open(path,"a" if append else "w",encoding='utf8')
	for l in list:
		f.write(str(l) + "\n")
	f.close()

def write_text(path, text, append = False):
    f = codecs.open(path,"a" if append else "w",encoding='utf8')
    f.write(text + "\n")
    f.close()

def load_csv_lines(path, delimiter = ',', indices = None):
	f = codecs.open(path,'r',encoding='utf8', errors = 'ignore')
	lines = [l.strip().split(delimiter) for l in f.readlines()]
	if indices is None:
		return lines
	else:
		return [sublist(l, indices) for l in lines if len(l) >= max(indices) + 1]

def sublist(list, indices):
	sublist = []
	for i in indices:	
		sublist.append(list[i])
	return sublist

### Annotations handling

def measure_correlations(path, indices):
    res = []
    lines = load_csv_lines(path, delimiter = '\t', indices = indices)[1:]
    for i in range(len(indices) - 1):
        for j in range(i + 1, len(indices)):
            vals1 = []
            for x in lines: 
              vals1.append(float(x[i])) #[float(x[i]) for x in lines]
            vals2 = [float(x[j]) for x in lines] 
            r = spearmanr(vals1, vals2)[0]
            r2 = pearsonr(vals1, vals2)[0] 
            res.append((i, j, r, r2))
    avg_spear = sum([x[2] for x in res]) / len(res)
    avg_pears = sum([x[3] for x in res]) / len(res)
    return res, avg_spear, avg_pears
    
### Results collecting
def hyper_search_best(path):
  res_files = [f for f in list(os.walk(path))[0][2] if f.endswith(".out.txt")]
  all_res = {}
  for rf in res_files:
    lines = [l for l in load_lines(path + "/" + rf) if l.startswith("HL ")]
    max_score = -1.0
    max_m = None
    for l in lines:
      m = l.replace(";", ":").split(":")[0].strip()
      score = float(l.replace(";", ":").split(":")[2].strip())
      if score > max_score:
        max_score = score
        max_m = m
    all_res[rf] = max_score
  all_res_sort = sorted(all_res.items(), key=lambda x: x[1])
  print(all_res_sort)
  return all_res
