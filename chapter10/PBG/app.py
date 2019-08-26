import json
import h5py
import os

DATA_DIR = "datas/soc-LiveJournal"
#加载生成的实体字典
with open(os.path.join(DATA_DIR,"dictionary.json"), "rt") as tf:
    dictionary = json.load(tf)
#查找用户"1"在字典中的索引
user_id = "0"
offset = dictionary["entities"]["user_id"].index(user_id)
print("用户{}在字典中的索引为{}".format(user_id, offset))

#加载嵌入文件
with h5py.File("model/demo/embeddings_user_id_1.v10.h5", "r") as hf:
    embedding_user_0 = hf["embeddings"][offset, :]
    embedding_all = hf["embeddings"][:]

print(embedding_user_0.shape)
print(embedding_all.shape)

from torchbiggraph.model import DotComparator
src_entity_offset = dictionary["entities"]["user_id"].index("0")
dest_1_entity_offset = dictionary["entities"]["user_id"].index("7")
dest_2_entity_offset = dictionary["entities"]["user_id"].index("135")

with h5py.File("model/demo/embeddings_user_id_0.v10.h5", "r") as hf:
    src_embedding = hf["embeddings"][src_entity_offset, :]
    dest_1_embedding = hf["embeddings"][dest_1_entity_offset, :]
    dest_2_embedding = hf["embeddings"][dest_2_entity_offset, :]
    dest_embeddings = hf["embeddings"][...]

import torch
comparator = DotComparator()

scores_1, _, _ = comparator(
    comparator.prepare(torch.tensor(src_embedding.reshape([1,1,520]))),
    comparator.prepare(torch.tensor(dest_1_embedding.reshape([1,1,520]))),
    torch.empty(1, 0, 520),  # Left-hand side negatives, not needed
    torch.empty(1, 0, 520),  # Right-hand side negatives, not needed
)

scores_2, _, _ = comparator(
    comparator.prepare(torch.tensor(src_embedding.reshape([1,1,520]))),
    comparator.prepare(torch.tensor(dest_2_embedding.reshape([1,1,520]))),
    torch.empty(1, 0, 520),  # Left-hand side negatives, not needed
    torch.empty(1, 0, 520),  # Right-hand side negatives, not needed
)

print(scores_1)
print(scores_2)
