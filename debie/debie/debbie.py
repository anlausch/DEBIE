import utils
import debiasing
import eval
import numpy as np
import weat
from sys import stdin
import sys

# CONFIG
lang = sys.argv[1]
lang_transfer = (lang != "en")
model = sys.argv[2]
test = int(sys.argv[3])

lowercase_stimuli = (model == "ft" or model == "glove") 
simlex_eval = lang in ["de", "hr", "ru", "it"]
wsim_eval = lang in ["de", "ru", "it"]

## WEAT 1
if test == 1:
  targets_1 = ["aster", "clover", "hyacinth", "marigold", "poppy", "azalea", "crocus", "iris", "orchid", "rose", "blue-bell", "daffodil", "lilac", "pansy", "tulip", "buttercup", "daisy", "lily", "peony", "violet", "carnation", "gladiola", "magnolia", "petunia", "zinnia"]
  targets_2 = ["ant", "caterpillar", "flea", "locust", "spider", "bedbug", "centipede", "fly", "maggot", "tarantula", "bee", "cockroach", "gnat", "mosquito", "termite", "beetle", "cricket", "hornet", "moth", "wasp", "blackfly", "dragonfly", "horsefly", "roach", "weevil"]
  attributes_1 = ["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow", "diploma", "gift", "honor", "miracle", "sunrise", "family", "happy", "laughter", "paradise", "vacation"]
  attributes_2 = ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", "grief", "poison", "stink", "assault", "disaster", "hatred", "pollute", "tragedy", "divorce", "jail", "poverty", "ugly", "cancer", "kill", "rotten", "vomit", "agony", "prison"]

  tgs_1_aug = ["scarlet", "bluebell", "cornflower", "delphinium", "fleabane", "amaranthus", "dianthus", "chromatic", "poof", "eonies", "orchidaceae", "orchis", "azaleas", "mauve", "tangerine", "nance", "tulipa", "camellia", "taupe", "willowherb", "hyacinths", "minaj", "periwinkle", "helianthemum", "poppies", "lilies", "cress", "magnolias", "macklemore", "dolly", "sissy", "sapphire", "orchids", "buddleja", "licorice", "jasmine", "faggot", "tulips", "lavender", "opium", "dandelion", "weeknd", "wisteria", "cowslip", "prunella", "thyme", "alfalfa", "lilacs", "daffodils", "magnoliaceae", "pinkish", "watercress", "crowfoot", "veronica", "primula", "carrie", "bluish", "cryptanthus", "trefoil", "asters", "jessie", "polly", "olive", "clovers", "meadowsweet", "fuchsia", "penstemon", "candlewood", "marigolds", "dandelions", "cyclamen", "snowberry", "purplish", "sassafras", "gladiolus", "epiphyte", "magenta"]
  tgs_2_aug = ["caterpillars", "wasps", "corsair", "whitefly", "insect", "bumblebee", "bowler", "noctuidae", "yellowjacket", "mayfly", "curculionidae", "cockroaches", "dragonflies", "avenger", "mulligan", "pilotless", "roundworm", "undershot", "protruding", "grasshopper", "crambidae", "damselfly", "louse", "projected", "cricketing", "vermin", "parasitoid", "tarantulas", "wicket", "sticking", "scorpion", "gnats", "hellcat", "mosquitoes", "sawfly", "hive", "arachnid", "larva", "locusts", "centipedes", "snook", "batsman", "weevils", "dart", "flit", "bug", "fleas", "gracillariidae", "harrier", "burrowing", "scamper", "roaches", "hickory", "mosquitos", "scoot", "tractor", "fathead", "worm", "bumblebees", "millipede", "pyralidae", "termites", "leafhopper", "ants"]

## WEAT 8
if test == 8:
  targets_1 = ["science", "technology", "physics", "chemistry", "Einstein", "NASA", "experiment", "astronomy"]
  targets_2 = ["poetry", "art", "Shakespeare", "dance", "literature", "novel", "symphony", "drama"]
  attributes_1 = ["brother", "father", "uncle", "grandfather", "son", "he", "his", "him"]
  attributes_2 = ["sister", "mother", "aunt", "grandmother", "daughter", "she", "hers", "her"]

  tgs_1_aug = ["physicists", "test", "electrochemistry", "automation", "engineering", "biophysics", "education", "learning", "chromodynamics", "technologies", "radiochemistry", "examination", "biology", "technological", "astronomer", "astrophysics", "experimentation", "biochemistry", "research", "lore", "electrodynamics", "astrobiology", "astrometry", "erudition"]
  tgs_2_aug = ["dramaturgy", "monograph", "untried", "dances", "poesy", "dissertation", "craftsmanship", "orchestra", "treatise", "skill", "waltz", "poem", "literatures", "dramatization", "poems", "theatre", "dancing", "newfound", "hop", "artistry", "new", "verse", "craft", "philharmonic", "concerto", "groundbreaking", "dramatics", "sinfonietta"]

### EN vectors ###

print("Loading vectors...")

# FT
if model == "ft":
  vecs, vecs_norm = utils.load_vectors("/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vectors", normalize = True)
  vocab, vocab_inv = utils.load_vocab("/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki.en.300.vocab", inverse = True)

# CBOW
elif model == "cbow":
  vecs, vecs_norm = utils.load_vectors("/work/anlausch/debbie/output/w2v_cbow_200k.vec", normalize = True)
  vocab, vocab_inv = utils.load_vocab("/work/anlausch/debbie/output/w2v_cbow_200k.vocab", inverse = True)

# GLOVE
elif model == "glove":
  vecs, vecs_norm = utils.load_vectors("/work/anlausch/debbie/output/glove_200k.vec", normalize = True)
  vocab, vocab_inv = utils.load_vocab("/work/anlausch/debbie/output/glove_200k.vocab", inverse = True)

print("Loading Anne's debiased vectors...")
# WEAT 8 model, FT
if model == "ft" and test == 8:
  vecs_anne, vecs_norm_anne = utils.load_vectors("/work/anlausch/debbie/output/data/weat_8_postspec_4_wo_original/drp=0.9_rf=0.2/" + (lang + "_debiased.vectors" if lang_transfer else "drp=0.9_rf=0.2.vec"), normalize = True)
  vecs_anne_srclang, vecs_norm_anne_srclang = utils.load_vectors("/work/anlausch/debbie/output/data/weat_8_postspec_4_wo_original/drp=0.9_rf=0.2/drp=0.9_rf=0.2.vec", normalize = True)

# WEAT 1 model, FT
if model == "ft" and test == 1:
  vecs_anne, vecs_norm_anne = utils.load_vectors("/work/anlausch/debbie/output/data/postspec_2_wo_original_repr/drp=0.9_rf=0.15/" + (lang + "_debiased.vectors" if lang_transfer else "drp=0.9_rf=0.15.vec"), normalize = True)
  vecs_anne_srclang, vecs_norm_anne_srclang = utils.load_vectors("/work/anlausch/debbie/output/data/postspec_2_wo_original_repr/drp=0.9_rf=0.15/drp=0.9_rf=0.15.vec", normalize = True)

# WEAT 8, CBOW 
if model == "cbow" and test == 8:
  vecs_anne, vecs_norm_anne = utils.load_vectors("/work/anlausch/debbie/output/data/cbow_weat_8_postspec_4_wo_original/drp=0.9_rf=0.1_ef=1.0_if=0.0/drp=0.9_rf=0.1_ef=1.0_if=0.0.vec", normalize = True)

# WEAT 1, CBOW
if model == "cbow" and test == 1:
  vecs_anne, vecs_norm_anne = utils.load_vectors("/work/anlausch/debbie/output/data/cbow_weat_1_postspec_2_wo_original/drp=0.9_rf=0.1_ef=1.0_if=0.0/drp=0.9_rf=0.1_ef=1.0_if=0.0.vec", normalize = True)

# WEAT 8, GLOVE
if model == "glove" and test == 8:
  vecs_anne, vecs_norm_anne = utils.load_vectors("/work/anlausch/debbie/output/data/glove_weat_8_postspec_4_wo_original/drp=0.9_rf=0.1_ef=1.0_if=0.0/drp=0.9_rf=0.1_ef=1.0_if=0.0.vec", normalize = True)

# WEAT 1, GLOVE
if model == "glove" and test == 1:
  vecs_anne, vecs_norm_anne = utils.load_vectors("/work/anlausch/debbie/output/data/glove_weat_1_postspec_2_wo_original/drp=0.9_rf=0.1_ef=1.0_if=0.0/drp=0.9_rf=0.1_ef=1.0_if=0.0.vec", normalize = True)



#/work/anlausch/debbie/output/data/glove_weat_8_postspec_4_wo_original
##########################

### Loading evaluation datasets, SimLex and WS353 ###

if simlex_eval:
  simlex = [(l.split("\t")[0].lower(), l.split("\t")[1].lower(), float(l.split("\t")[3 if not lang_transfer else (2 if lang == "hr" else 3)])) for l in utils.load_lines("/work/gglavas/data/evaluation/simlex999/" + (lang.upper() + "-" if lang_transfer else "") + "SimLex-999.txt")]
#simlex_clean = [(l.split("\t")[0], l.split("\t")[1], float(l.split("\t")[3])) for l in utils.load_lines("/work/anlausch/SimLex-999/SimLex-GN-light.txt")[1:]]

#wsim = [(l.split("\t")[1], l.split("\t")[2], float(l.split("\t")[3])) for l in utils.load_lines("/work/gglavas/data/evaluation/wsim353/wsim.txt")]
if wsim_eval:
  wsim = [(l.split("," if lang_transfer else "\t")[0 if lang_transfer else 1].lower(), l.split("," if lang_transfer else "\t")[1 if lang_transfer else 2].lower(), float(l.split("," if lang_transfer else "\t")[-1 if lang_transfer else 3])) for l in utils.load_lines("/work/gglavas/data/evaluation/wsim353/" + ("ws353-" + lang if lang_transfer else "wsim") + ".txt")[1:]]

##################################### transformation / debiasing methods #################################


### XWEAT translations ###

if lang_transfer:
  trans_dict = np.load("/work/anlausch/xweat/data/vocab_dict_en_" + lang + ".p")

  vecs_tgt_bp, vecs_norm_tgt_bp = utils.load_vectors("/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki." + lang + ".300.vectors", normalize = True)
  vocab_tgt_bp, vocab_inv_tgt_bp = utils.load_vocab("/work/gglavas/data/word_embs/yacle/fasttext/200K/npformat/ft.wiki." + lang + ".300.vocab", inverse = True)

  vecs_tgt, vecs_tgt_norm  = utils.load_vectors("/work/gglavas/data/word_embs/yacle/mappings/new/smith/fasttext/" + lang + "-en/" + ((lang + "-en." + lang + ".yacle.train.freq.5k.vectors") if lang in ["de", "ru", "tr"] else (lang + ".vectors")), normalize = True)
  vocab_tgt, vocab_tgt_inv = utils.load_vocab("/work/gglavas/data/word_embs/yacle/mappings/new/smith/fasttext/" + lang + "-en/" + ((lang + "-en." + lang + ".yacle.train.freq.5k.vocab") if lang in ["de", "ru", "tr"] else (lang + ".vocab")), inverse = True)

  if test == 8:
    lines = [list(x.split(":")[1].strip().split()) for x in utils.load_lines("/work/anlausch/xweat/data/weat_8_" + lang + ".txt")]
  elif test == 1:
    lines = [list(x.split(":")[1].strip().split()) for x in utils.load_lines("/work/anlausch/xweat/data/weat_1_" + lang + ".txt")]

  targets_1_tgt = [x.strip() for x in lines[0]]
  targets_2_tgt = [x.strip() for x in lines[1]] 
  attributes_1_tgt =  [x.strip() for x in lines[2]] 
  attributes_2_tgt =  [x.strip() for x in lines[3]] 

  if lowercase_stimuli:
    targets_1_tgt = [x.lower() for x in targets_1_tgt]
    targets_2_tgt = [x.lower() for x in targets_2_tgt]  
    attributes_1_tgt = [x.lower() for x in attributes_1_tgt]  
    attributes_2_tgt = [x.lower() for x in attributes_2_tgt] 

  print(targets_1_tgt)
  print(targets_2_tgt)
  print(attributes_1_tgt)
  print(attributes_2_tgt)

#stdin.readline()


### Training instances ###
eq_pairs = [] 
for t1 in tgs_1_aug:
  for t2 in tgs_2_aug:
    eq_pairs.append((t1, t2))

### Transformation
v_b = debiasing.get_bias_direction(eq_pairs, vecs, vecs_norm, vocab, vocab_inv)
utah = debiasing.debias_direction_linear(v_b, vecs_norm)

if not lang_transfer:
  v_b_anne = debiasing.get_bias_direction(eq_pairs, vecs_anne, vecs_norm_anne, vocab, vocab_inv)
  utah_anne = debiasing.debias_direction_linear(v_b_anne, vecs_norm_anne)
else:
  v_b_anne = debiasing.get_bias_direction(eq_pairs, vecs_anne_srclang, vecs_norm_anne_srclang, vocab, vocab_inv)
  utah_anne = debiasing.debias_direction_linear(v_b_anne, vecs_norm_anne)
  
proc, proc_proj_mat = debiasing.debias_proc(eq_pairs, vecs, vocab)

v_b_proc = debiasing.get_bias_direction(eq_pairs, proc, proc, vocab, vocab_inv)
proc_utah = debiasing.debias_direction_linear(v_b_proc, proc)

utah_proc, utah_proc_proj_mat = debiasing.debias_proc(eq_pairs, utah, vocab)

### language transfer
if lang_transfer:
  targets_1 = targets_1_tgt
  targets_2 = targets_2_tgt
  attributes_1 = attributes_1_tgt
  attributes_2 = attributes_2_tgt

  utah_tgt = debiasing.debias_direction_linear(v_b, vecs_tgt_norm)

  proc_tgt = np.matmul(vecs_tgt, proc_proj_mat)
  proc_tgt = (proc_tgt + vecs_tgt) / 2

  proc_utah_tgt = debiasing.debias_direction_linear(v_b_proc, proc_tgt)
  
  utah_proc_tgt = np.matmul(utah_tgt, utah_proc_proj_mat)
  utah_proc_tgt = (utah_proc_tgt + utah_tgt) / 2

  method_dict = method_dict = {"distributional" : vecs_tgt_bp, "utah" : utah_tgt, "procrustes" : proc_tgt, "anne" : vecs_anne, "utah-anne" : utah_anne, "procrustes-utah" : proc_utah_tgt, "utah-procrustes" : utah_proc_tgt}
  method_train_dict = {"distributional" : vecs, "utah" : utah, "anne" : vecs_anne, "utah-anne" : utah_anne, "procrustes" : proc, "utah-procrustes" : utah_proc, "procrustes-utah" : proc_utah}

  vocab_train_svm = vocab
  vocab = vocab_tgt

  #missing = [x for x in targets_1 if x.lower() not in vocab] + [x for x in targets_2 if x.lower() not in vocab]
  #print(missing)
  #stdin.readline()

else:
  method_dict = {"distributional" : vecs, "utah" : utah, "procrustes" : proc, "anne" : vecs_anne, "utah-anne" : utah_anne, "procrustes-utah" : proc_utah, "utah-procrustes" : utah_proc}

if lowercase_stimuli:
  targets_1 = [x.lower() for x in targets_1]
  targets_2 = [x.lower() for x in targets_2]  
  attributes_1 = [x.lower() for x in attributes_1]  
  attributes_2 = [x.lower() for x in attributes_2]  

### Output debiased spaces ###
#destination_path = "/work/gglavas/data/word_embs/debie/"
#for m in method_dict:
#  vectors = method_dict[m]
#  np.save(destination_path + m + ".vectors", vectors)
#exit()  

##########################################################################################################

print("### SVM ###")

for m in method_dict:
  vectors = method_dict[m]
  if lang_transfer:
    vecs_train = method_train_dict[m]
  acc = eval.eval_svm(tgs_1_aug, tgs_2_aug, targets_1, targets_2, vecs_train if lang_transfer else vectors, vectors, vocab_train_svm if lang_transfer else vocab, vocab_tgt if lang_transfer else vocab)
  print(m)
  print(acc)
  print()

print("### BAT ###")

for m in method_dict:
  vectors = method_dict[m]
  mrr = eval.bias_analogy_test(vectors, vocab, targets_1, targets_2, attributes_1, attributes_2)  
  print(m)
  print(mrr)
  print()

stdin.readline()

print("### WEAT ###")

for m in method_dict:
  vectors = method_dict[m]
  w = weat.XWEAT(300)
  w.embedding_matrix = vectors
  w.vocab = vocab
  #print("Computing similarities...")
  #w._init_similarities("cosine")
  #print("To compute WEAT test")
  test_stat, effect, p = w.run_test_precomputed_sims(targets_1, targets_2, attributes_1, attributes_2, sample_p = 5000)
  print(m)
  print(effect, p)
  print()

#stdin.readline()

print("### KMeans++ ###")

for m in method_dict:
  vectors = method_dict[m]
  acc = eval.eval_k_means(targets_1, targets_2, vectors, vocab)
  print(m)
  print(acc)
  print() 

#stdin.readline()

print("### ECT ###")

for m in method_dict:
  vectors = method_dict[m]
  spear = eval.embedding_coherence_test(vectors, vocab, targets_1, targets_2, attributes_2 + attributes_1)
  print(m)
  print(spear)
  print()

#stdin.readline()

if simlex_eval:
  print("### Regular SimLex ###")

  for m in method_dict:
    vectors = method_dict[m]
    p, s = eval.eval_simlex(simlex, vocab, vectors)
    print(m)
    print(p, s)
    print()

  #stdin.readline()

  #print("### Clean SimLex ###")

  #for m in method_dict:
  #  vectors = method_dict[m]
  #  p, s = eval.eval_simlex(simlex_clean, vocab, vectors)
  #  print(m)
  #  print(p, s)
  #  print()

  #stdin.readline()

if wsim_eval:
  print("### WSIM-353 ###")

  for m in method_dict:
    vectors = method_dict[m]
    p, s = eval.eval_simlex(wsim, vocab, vectors)
    print(m)
    print(p, s)
    print()

 # stdin.readline()



