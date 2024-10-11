import os
import pickle
import requests
import tiktoken
import numpy as np
from datasets import load_dataset

# ds = load_dataset("EleutherAI/arithmetic", "arithmetic_1dc")

# data = ["{}{}".format(_["context"], _["completion"]) for _ in ds["validation"]]
# print(data[0])
# data = "\n".join(data)

with open(os.path.join(os.path.dirname(__file__), "addition.txt")) as f:
    data = f.readlines()
    data = [_ for _ in data]

chars = sorted(list(set("\n".join(data))))
print(chars)
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print(itos)


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    return "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string


n = len(data)

train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

train_ids = [np.array(encode(_)) for _ in train_data]
val_ids = [np.array(encode(_)) for _ in val_data]

# val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} examples")
print(f"val has {len(val_ids):,} examples")

# export to bin files
train_ids = np.array(train_ids, dtype=object)
val_ids = np.array(val_ids, dtype=object)
np.save(os.path.join(os.path.dirname(__file__), "train"), train_ids)
np.save(os.path.join(os.path.dirname(__file__), "val"), val_ids)

# save the meta information as well, to help us encode/decode later
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
