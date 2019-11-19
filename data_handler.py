import codecs
from itertools import product
import pickle
from util import fuse_stimuli
from util import flatten
from util import read_weat_data
from util import select_random_attributes

class InputExample(object):

  def __init__(self, guid, t1=None, t2=None, a=None, label=None):
    """Constructs a InputExample."""
    self.guid = guid
    self.t1 = t1
    self.t2 = t2
    self.a = a
    self.label = label


def output_weat_dict_to_file(weat_dict, output_path):
  with codecs.open(output_path, "w", "utf8") as f:
    for key, value in weat_dict.items():
      f.write(key + " ")
      f.write(' '.join(value))
      f.write('\n')


def split_weat_in_train_dev(input_paths, output_train, output_dev):
  """
  :param input_paths:
  :param output_train:
  :param output_dev:
  :return:
  >>> split_weat_in_train_dev(["./data/weat_1_aug_postspec_2_new.txt"], "./data/weat_1_aug_postspec_2_train.txt", "./data/weat_1_aug_postspec_2_dev.txt")
  """
  t1, t2, a1, a2 = fuse_stimuli(input_paths)
  weat_dict = {}
  weat_dict["T1:"] = t1
  weat_dict["T2:"] = t2
  weat_dict["A1:"] = a1
  weat_dict["A2:"] = a2
  train_dict = {}
  dev_dict = {}
  for key, value in weat_dict.items():
    split_index = int(len(value) * 0.7)
    train_dict[key] = value[:split_index]
    dev_dict[key] = value[split_index:]
  output_weat_dict_to_file(train_dict, output_train)
  output_weat_dict_to_file(dev_dict, output_dev)



def filter_vocabulary(term_list, vocab_path="/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab"):
  vocab = pickle.load(open(vocab_path, "rb"))
  new_term_list = []
  for t in term_list:
    if t in vocab:
      new_term_list.append(t)
    else:
      print("Not in vocab %s" % t)
  return new_term_list


def prepare_input_examples(input_paths, output_path, random_attributes=False,
                           sampling_vocab_path="/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab",
                           original_weat_path="", k=60, switch_targets_and_attributes=False):
  """
  :param paths:
  :return:
  prepare_input_examples(["./data/weat_1.txt", "./data/weat_1_augmentation.txt"], "./data/weat_1_prepared_filtered.txt")
  prepare_input_examples(["./data/weat_1.txt", "./data/weat_1_augmentation_filtered_cleaned.txt"], "./data/weat_1_prepared_filtered_cleaned.txt")
  prepare_input_examples(["./data/weat_1_augmentation_filtered_cleaned.txt"], "./data/weat_1_prepared_filtered_cleaned_wo_original.txt")
  prepare_input_examples(["./data/weat_1_augmentation.txt"], "./data/weat_1_prepared_filtered_wo_original.txt")
  #>>> prepare_input_examples(["./data/weat_1_aug_postspec_2_new.txt"], "./data/weat_1_prepared_filtered_postspec_2_wo_original.txt")
  #>>> prepare_input_examples(["./data/weat_1.txt","./data/weat_1_aug_postspec_2_new.txt"], "./data/weat_1_prepared_filtered_postspec_2.txt")
  #>>> prepare_input_examples(["./data/weat_1.txt","./data/weat_1_aug_postspec_3_new.txt"], "./data/weat_1_prepared_filtered_postspec_3.txt")
  #>>> prepare_input_examples(["./data/weat_1.txt","./data/weat_1_aug_postspec_4_new.txt"], "./data/weat_1_prepared_filtered_postspec_4.txt")
  #>>> prepare_input_examples(["./data/weat_1.txt","./data/weat_1_aug_postspec_5_new.txt"], "./data/weat_1_prepared_filtered_postspec_5.txt")
  #>>> prepare_input_examples(["./data/weat_1_aug_postspec_3_new.txt"], "./data/weat_1_prepared_filtered_postspec_3_wo_original.txt")
  #>>> prepare_input_examples(["./data/weat_1_aug_postspec_4_new.txt"], "./data/weat_1_prepared_filtered_postspec_4_wo_original.txt")
  #>>> prepare_input_examples(["./data/weat_1_aug_postspec_5_new.txt"], "./data/weat_1_prepared_filtered_postspec_5_wo_original.txt")
  #>>> prepare_input_examples(["./data/weat_1_aug_postspec_2_train.txt"], "./data/weat_1_prepared_filtered_postspec_2_wo_original_train.txt")
  #>>> prepare_input_examples(["./data/weat_1_aug_postspec_2_dev.txt"], "./data/weat_1_prepared_filtered_postspec_2_wo_original_dev.txt")
  #>>> prepare_input_examples(["./data/weat_8_aug_postspec_2_new.txt"], "./data/weat_8_prepared_filtered_postspec_2_wo_original.txt")
  #>>> prepare_input_examples(["./data/weat_8_aug_postspec_4_new.txt"], "./data/weat_8_prepared_filtered_postspec_4_wo_original.txt")
  #>>> prepare_input_examples(["./data/weat_8_aug_postspec_random_4_switched=False.txt"], "./data/weat_8_prepared_filtered_postspec_4_wo_original_random_switched=False.txt")
  #>>> prepare_input_examples(["./data/weat_8_aug_postspec_random_4_switched=True.txt"], "./data/weat_8_prepared_filtered_postspec_4_wo_original_random_switched=True.txt")
  #>>> prepare_input_examples(["./data/weat_8_aug_postspec_4_new.txt"], "./data/weat_8_prepared_filtered_postspec_4_wo_original_random.txt", \
  random_attributes=True, original_weat_path="./data/weat_8.txt", k=60, switch_targets_and_attributes=False)
  #>>> prepare_input_examples(["./data/weat_8_aug_postspec_4_new.txt"], "./data/weat_8_prepared_filtered_postspec_4_wo_original_random_switched.txt", \
  random_attributes=True, original_weat_path="./data/weat_8.txt", k=60, switch_targets_and_attributes=True)
  >>> prepare_input_examples(["./data/weat_8_aug_postspec_4_new.txt"], "./data/weat_8_prepared_filtered_postspec_4_wo_original_switched.txt", switch_targets_and_attributes=True)
  """
  t1, t2, a1, a2 = fuse_stimuli(input_paths)
  if switch_targets_and_attributes:
    t1, a1 = a1, t1
    t2, a2 = a2, t2
  print("Number of terms in t1 %d" % len(t1))
  print("Number of terms in t2 %d" % len(t2))
  t1 = filter_vocabulary(t1)
  t2 = filter_vocabulary(t2)
  print("Number of terms in t1 after filtering %d" % len(t1))
  print("Number of terms in t2 after filtering %d" % len(t2))

  if not random_attributes:
    a = flatten([a1, a2])
    print("Number of terms in a %d" % len(a))
    a = filter_vocabulary(a)
    print("Number of terms in a after filtering %d" % len(a))
    with codecs.open(output_path, "w", "utf8") as f:
      for comb in product(t1, t2, a):
        f.write(comb[0] + "\t" + comb[1] + "\t" + comb[2] + "\n")
      f.close()
  else:
    vocab = pickle.load(open(sampling_vocab_path, "rb"))
    # transform vocab dict to list by the order of the keys:
    unordered_vocab_list = [(term, index) for (term, index) in vocab.items()]
    ordered_vocab_list = sorted(unordered_vocab_list, key=lambda x: x[1])
    vocab_list = [term for (term, index) in ordered_vocab_list]
    with codecs.open(output_path, "w", "utf8") as f:
      for comb in product(t1, t2):
        attributes = select_random_attributes(vocab_list, k, original_weat_path)
        for a in attributes:
          f.write(comb[0] + "\t" + comb[1] + "\t" + a + "\n")
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




