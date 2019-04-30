import codecs
from itertools import product
import pickle

class InputExample(object):

  def __init__(self, guid, t1=None, t2=None, a=None, label=None):
    """Constructs a InputExample."""
    self.guid = guid
    self.t1 = t1
    self.t2 = t2
    self.a = a
    self.label = label


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


def filter_vocabulary(term_list, vocab_path="/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab"):
  vocab = pickle.load(open(vocab_path, "rb"))
  new_term_list = []
  for t in term_list:
    if t in vocab:
      new_term_list.append(t)
    else:
      print("Not in vocab %s" % t)
  return new_term_list


def prepare_input_examples(input_paths, output_path):
  """
  :param paths:
  :return:
  prepare_input_examples(["./data/weat_1.txt", "./data/weat_1_augmentation.txt"], "./data/weat_1_prepared_filtered.txt")
  prepare_input_examples(["./data/weat_1.txt", "./data/weat_1_augmentation_filtered_cleaned.txt"], "./data/weat_1_prepared_filtered_cleaned.txt")
  prepare_input_examples(["./data/weat_1_augmentation_filtered_cleaned.txt"], "./data/weat_1_prepared_filtered_cleaned_wo_original.txt")
  prepare_input_examples(["./data/weat_1_augmentation.txt"], "./data/weat_1_prepared_filtered_wo_original.txt")
  #>>> prepare_input_examples(["./data/weat_1_aug_postspec_2.txt"], "./data/weat_1_prepared_filtered_postspec_2_wo_original.txt")
  #>>> prepare_input_examples(["./data/weat_1.txt","./data/weat_1_aug_postspec_2.txt"], "./data/weat_1_prepared_filtered_postspec_2.txt")
  >>> prepare_input_examples(["./data/weat_1.txt","./data/weat_1_aug_postspec_3.txt"], "./data/weat_1_prepared_filtered_postspec_3.txt")
  >>> prepare_input_examples(["./data/weat_1.txt","./data/weat_1_aug_postspec_4.txt"], "./data/weat_1_prepared_filtered_postspec_4.txt")
  >>> prepare_input_examples(["./data/weat_1.txt","./data/weat_1_aug_postspec_5.txt"], "./data/weat_1_prepared_filtered_postspec_5.txt")
  >>> prepare_input_examples(["./data/weat_1_aug_postspec_3.txt"], "./data/weat_1_prepared_filtered_postspec_3_wo_original.txt")
  >>> prepare_input_examples(["./data/weat_1_aug_postspec_4.txt"], "./data/weat_1_prepared_filtered_postspec_4_wo_original.txt")
  >>> prepare_input_examples(["./data/weat_1_aug_postspec_5.txt"], "./data/weat_1_prepared_filtered_postspec_5_wo_original.txt")
  """
  t1, t2, a1, a2 = fuse_stimuli(input_paths)
  a = flatten([a1, a2])
  print("Number of terms in t1 %d" % len(t1))
  print("Number of terms in t2 %d" % len(t2))
  print("Number of terms in a %d" % len(a))
  t1 = filter_vocabulary(t1)
  t2 = filter_vocabulary(t2)
  a = filter_vocabulary(a)
  print("Number of terms in t1 after filtering %d" % len(t1))
  print("Number of terms in t2 after filtering %d" % len(t2))
  print("Number of terms in a after filtering %d" % len(a))
  with codecs.open(output_path, "w", "utf8") as f:
    for comb in product(t1, t2, a):
      f.write(comb[0] + "\t" + comb[1] + "\t" + comb[2] + "\n")
    f.close()


def load_input_examples(path):
  examples = []
  with codecs.open(path, "r", "utf8") as f:
    for i, line in enumerate(f.readlines()):
      parts = line.strip().split("\t")
      examples.append(InputExample(guid=i, t1=parts[0], t2=parts[1], a=parts[2]))
  return examples


def clean_lists(list_path, output_path="./data/weat_1_augmentation_filtered.txt"):
  """
  :param list_path:
  :param output_path:
  :return:
   >>> clean_lists("./data/weat_9_augmentation.txt", "./data/weat_9_augmentation_filtered.txt")
  """
  weat_dict = read_weat_data(list_path)
  with codecs.open(output_path, "w", "utf8") as f:
    for key, value in weat_dict.items():
      new_list = filter_vocabulary(value)
      f.write(key + " ")
      f.write(' '.join(new_list))
      f.write('\n')




