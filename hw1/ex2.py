
#Name: Ankita Patra
#B:Number: B01101280
#Mail id : apatra@binghamton.edu

# ex2.py — SST-2 embeddings + Logistic Regression 

import os
# avoid tokenizer and fork warning in terminal
os.environ["TOKENIZERS_PARALLELISM"] = "false"   
import time
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

t0 = time.time()

# 1) Load SST-2 (train + dev)
ds = load_dataset("stanfordnlp/sst2")
train_texts = ds["train"]["sentence"]
y_train     = np.array(ds["train"]["label"], dtype=np.int64)
dev_texts   = ds["validation"]["sentence"]
y_dev       = np.array(ds["validation"]["label"], dtype=np.int64)

# 2) Pretrained sentence encoder (defaults to CPU)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode → embeddings
X_train = model.encode(train_texts, batch_size=256, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
X_dev   = model.encode(dev_texts,   batch_size=256, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)

# 3) Train Logistic Regression on embeddings
clf = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=1, random_state=42)
clf.fit(X_train, y_train)

# Evaluate on dev
dev_acc = accuracy_score(y_dev, clf.predict(X_dev))
print(f"Dev Accuracy: {dev_acc*100:.2f}%")
print(f"Done in {time.time() - t0:.1f}s.")


"""
o/p :
(.venv) ankitapatra@Ankitas-MacBook-Pro hw1 % python ex2.py
Batches: 100%|█████████████████████████████████████████████| 264/264 [00:14<00:00, 17.89it/s]
Batches: 100%|█████████████████████████████████████████████████| 4/4 [00:00<00:00, 11.36it/s]
Dev Accuracy: 81.08%
Done in 21.3s.

"""
