import codecs
import os
from scipy.stats import spearmanr
from scipy.stats import pearsonr


### RAW IO

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
