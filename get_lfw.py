import os
from sklearn.datasets import fetch_lfw_pairs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
lfw = fetch_lfw_pairs(subset="10_folds", data_home=os.path.join(BASE_DIR, "data", "lfw"))
print("LFW downloaded:", lfw.pairs.shape)