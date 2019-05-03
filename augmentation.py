'''
Data augmentation with specialized embedding space
'''
import codecs
import numpy as np
import data_handler
import argparse

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
    return embbedding_dict, vocab_list, vector_list, word2index


def mat_normalize(mat, norm_order=2, axis=1):
  return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])



def cosine(a, b):
  norm_a = mat_normalize(a)
  norm_b = mat_normalize(b)
  cos = np.dot(norm_a, np.transpose(norm_b))
  return cos


def get_top_k_similar(term, similarity_matrix, word2index, vocab_list, k, test_vocab):
  index = word2index[term]
  similarities = similarity_matrix[index]
  top_k_indices = similarities.argsort()[-(k+len(test_vocab)):][::-1]
  top_k_terms = []
  for i in top_k_indices:
    if i != index and vocab_list[i] not in test_vocab:
      top_k_terms.append(vocab_list[i])
    if len(top_k_terms) == k:
      break
  return top_k_terms


def augment_term_list(original_list, similarity_matrix, word2index, vocab_list, k, test_vocab):
  augmented_list = []
  for term in original_list:
    if term in word2index:
      top_k = get_top_k_similar(term, similarity_matrix, word2index, vocab_list, k, test_vocab)
      augmented_list.append(top_k)
    else:
      print("Not in vocab: %s" % term)
  augmented_list = list(set(data_handler.flatten(augmented_list)))
  return augmented_list



def augment_weat_lists(path_to_weat="./data/weat_1.txt", path_to_embeddings="/home/anlausch/post-specialized-embeddings/postspec/ft_postspec.txt", k=2, output_path=""):
  t1, t2, a1, a2 = data_handler.fuse_stimuli([path_to_weat])
  embbedding_dict, vocab_list, vector_list, word2index = load_specialized_embeddings(path_to_embeddings)
  similarity_matrix = cosine(vector_list, vector_list)
  weat_dict = {}
  test_vocab = list(set(t1 + t2 + a1 + a2))
  weat_dict["T1:"] = augment_term_list(t1, similarity_matrix, word2index, vocab_list, k, test_vocab)
  weat_dict["T2:"] = augment_term_list(t2, similarity_matrix, word2index, vocab_list, k, test_vocab)
  weat_dict["A1:"] = augment_term_list(a1, similarity_matrix, word2index, vocab_list, k, test_vocab)
  weat_dict["A2:"] = augment_term_list(a2, similarity_matrix, word2index, vocab_list, k, test_vocab)
  print("Length of augmentation for T1: %s" % str(len(weat_dict["T1:"])))
  print("Length of augmentation for T2: %s" % str(len(weat_dict["T2:"])))
  print("Length of augmentation for A1: %s" % str(len(weat_dict["A1:"])))
  print("Length of augmentation for A2: %s" % str(len(weat_dict["A2:"])))
  with codecs.open(output_path, "w", "utf8") as f:
    for key, value in weat_dict.items():
      f.write(key + " ")
      f.write(' '.join(value))
      f.write('\n')
    f.close()


def test_vocab_overlap(path_1, path_2):
  """
  :param path_1:
  :param path_2:
  :return:
  >>> test_vocab_overlap("./data/weat_8.txt","./data/weat_8_aug_postspec_5_new.txt")
  """
  l1_t1, l1_t2, l1_a1, l1_a2 = data_handler.fuse_stimuli([path_1])
  l2_t1, l2_t2, l2_a1, l2_a2 = data_handler.fuse_stimuli([path_2])
  l1 = list(set(l1_t1 + l1_t2 + l1_a1 + l1_a2))
  l2 = list(set(l2_t1 + l2_t2 + l2_a1 + l2_a2))
  print(list(set(l1).intersection(l2)))


def main():
  parser = argparse.ArgumentParser(description="Running DEBIE's Augmentation")
  parser.add_argument("--path_to_weat", type=str, help="Path to the WEAT input file", required=True)
  parser.add_argument("--path_to_embeddings", type=str, default=None,
                      help="Path to the embedding files to augment with", required=True)
  parser.add_argument("--k", type=int, default=None,
                      help="Top k neighbors to augment with", required=True)
  parser.add_argument("--output_path", type=str, default=None,
                      help="Output path", required=True)

  args = parser.parse_args()


  augment_weat_lists(args.path_to_weat, args.path_to_embeddings, args.k, args.output_path)

if __name__ == "__main__":
  main()