#import matplotlib.rcsetup
#font = {'family': 'sans-serif',
#        'weight': 2,
#        'size': 0.5}
#matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
import numpy as np
import pickle
from data_handler import fuse_stimuli
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

def pca_plot(embedding_vector_path, embedding_vocab_path, vocab_to_plot, output_path):
  "Creates and TSNE model and plots it"
  sns.set()
  sns.set_style("whitegrid")
  sns.set_context("paper", rc={"font.size": 6})

  #fig, ax = plt.subplots(figsize=(16, 14))
  ax= plt.axes()
  #ax.set_ymargin
  ax.yaxis.grid(True, linestyle="dotted")
  ax.xaxis.grid(True, linestyle="dotted")
  ax.set_yticklabels([])
  ax.set_xticklabels([])

  vectors = np.load(embedding_vector_path)
  vocab = pickle.load(open(embedding_vocab_path, "rb"))
  labels = []
  tokens = []
  ids = []
  for id,group in enumerate(vocab_to_plot):
    for term in group:
      if term in vocab:
        ids.append(id)
        labels.append(term)
        tokens.append(vectors[vocab[term]])
  pca = PCA(n_components=2)
  new_values = pca.fit_transform(tokens)

  x = []
  y = []
  for value in new_values:
    x.append(value[0])
    y.append(value[1])

  for i in range(len(x)):
    if ids[i] == 0:
      plt.scatter(x[i], y[i], color=sns.color_palette()[0])
    elif ids[i] == 1:
      plt.scatter(x[i], y[i], color=sns.color_palette()[1], marker="s")
    elif ids[i] == 2:
      plt.scatter(x[i], y[i], color=sns.color_palette()[2], marker="D")
    elif ids[i] == 3:
      plt.scatter(x[i], y[i], color=sns.color_palette()[3],marker="^")
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  #ax.set_xlim([-3.0, 3.0])
  #ax.set_ylim([-3.0, 3.0])
  plt.savefig(output_path)#, bbox_inches='tight')

def pca_plot3d(embedding_vector_path, embedding_vocab_path, vocab_to_plot, output_path):
  # This import registers the 3D projection, but is otherwise unused.
  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
  from mpl_toolkits.mplot3d import proj3d
  "Creates and TSNE model and plots it"
  sns.set()
  ax = plt.figure().add_subplot(111, projection='3d')
  #ax= plt.axes()
  #ax.set_ymargin
  ax.yaxis.grid(True, linestyle="dotted")
  ax.xaxis.grid(True, linestyle="dotted")
  ax.zaxis.grid(True, linestyle="dotted")
  with sns.plotting_context("paper"):
    vectors = np.load(embedding_vector_path)
    vocab = pickle.load(open(embedding_vocab_path, "rb"))
    x = []
    y = []
    z = []
    ids = []
    labels = []
    for id,group in enumerate(vocab_to_plot):
      tokens = []
      for term in group:
        if term in vocab:
          labels.append(term)
          tokens.append(vectors[vocab[term]])
      pca = PCA(n_components=3)
      new_values = pca.fit_transform(tokens)


      for value in new_values:
        x.append(value[0])
        y.append(value[1])
        z.append(value[2])
      ids.append(id)


    for i in range(len(x)):
      if id == 0:
        plt.scatter(x[i], y[i], z[i], color=sns.color_palette()[0])
      elif id == 1:
        plt.scatter(x[i], y[i], z[i], color=sns.color_palette()[1])
      elif id == 2:
        plt.scatter(x[i], y[i], z[i],  color=sns.color_palette()[2])
      elif id == 3:
        plt.scatter(x[i], y[i], z[i],  color=sns.color_palette()[3])
      x2, y2, _ = proj3d.proj_transform(x[i], y[i], z[i], ax.get_proj())
      ax.annotate(labels[i],
                   xy=(x2, y2),
                   xytext=(5, 2),
                   textcoords='offset points',
                   ha='right',
                   va='bottom')

    #ax.set_xlim([-3.0, 3.0])
    #ax.set_ylim([-3.0, 3.0])
    plt.savefig(output_path)


def pca_plot_2(embedding_vector_path, embedding_vocab_path, vocab_to_plot, output_path):
  "Creates and TSNE model and plots it"
  sns.set(style="ticks", color_codes=True)
  with sns.plotting_context("paper"):
    vectors = np.load(embedding_vector_path)
    vocab = pickle.load(open(embedding_vocab_path, "rb"))
    #plt.figure(figsize=(16, 16))
    data = []
    tokens = []
    for id,group in enumerate(vocab_to_plot):
      for term in group:
        if term in vocab:
          data.append({'term':term, 'token': vectors[vocab[term]], 'group': id})
          tokens.append(vectors[vocab[term]])
    pca = PCA(n_components=2)
    new_values = pca.fit_transform(tokens)
    for i, d in enumerate(data):
      d['vector_x'] = new_values[i][0]
      d['vector_y'] = new_values[i][1]


    df = pd.DataFrame(data)
    sns.catplot(x="vector_x", y="vector_y", hue="group", data=df)
    for index, row in df.iterrows():
      plt.annotate(row["term"],
                   xy=(row["vector_x"], row["vector_y"]),
                   ha='right',
                   va='bottom')
    plt.savefig(output_path)


def plot_weat_vocab(embedding_vector_path, embedding_vocab_path, weat_input_path, output_path):
  """
  :param embedding_vector_path:
  :param embedding_vocab_path:
  :param weat_input_path:
  :param output_path:
  :return:
   >>> plot_weat_vocab("/work/gglavas/data/word_embs/debie/distributional.vectors.npy", "/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab", "./data/weat_8.txt", "./output/plots/distributional.pdf")
   #>>> plot_weat_vocab("/work/gglavas/data/word_embs/debie/utah-procrustes.vectors.npy", "/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab", "./data/weat_8.txt", "./output/plots/utah_procrustes.pdf")
   #>>> plot_weat_vocab("/work/gglavas/data/word_embs/debie/procrustes-utah.vectors.npy", "/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab", "./data/weat_8.txt", "./output/plots/procrustes_utah.pdf")
   #>>> plot_weat_vocab("/work/gglavas/data/word_embs/debie/utah.vectors.npy", "/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab", "./data/weat_8.txt", "./output/plots/utah.pdf")
   #>>> plot_weat_vocab("/work/gglavas/data/word_embs/debie/procrustes.vectors.npy", "/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab", "./data/weat_8.txt", "./output/plots/procrustes.pdf")
   #>>> plot_weat_vocab("/work/gglavas/data/word_embs/debie/anne.vectors.npy", "/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab", "./data/weat_8.txt", "./output/plots/anne.pdf")
   #>>> plot_weat_vocab("/work/gglavas/data/word_embs/debie/utah-anne.vectors.npy", "/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab", "./data/weat_8.txt", "./output/plots/utah-anne.pdf")
  """
  t1, t2, a1, a2 = fuse_stimuli([weat_input_path])
  #vocab_to_plot = t1 + t2 + a1 + a2
  pca_plot(embedding_vector_path, embedding_vocab_path, [t1, t2, a1, a2], output_path)