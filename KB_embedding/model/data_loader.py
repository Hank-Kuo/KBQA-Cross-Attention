import os 
from collections import Counter
from torch.utils import data
from typing import Dict, Tuple

Mapping = Dict[str, int]

def load_data(dataset_path, reverse=False):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = f.read().strip().split("\n")
        data = [i.split('\t') for i in data]
        if reverse:
            data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
    return data

def create_mappings(train_data, valid_data, test_data, output_dir) -> Tuple[Mapping, Mapping]:
    """Creates separate mappings to indices for entities and relations."""
    data = train_data + valid_data + test_data
    entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
    relations = sorted(list(set([d[1] for d in data])))

    entity2id = {entities[i]:i for i in range(len(entities))}
    relation2id = {relations[i]:i for i in range(len(relations))}
    
    entity_dic_path = os.path.join(output_dir, "entities.dict")
    relation_dic_path = os.path.join(output_dir, "relations.dict")

    print("Save entities and relations to dict")

    f = open(entity_dic_path, 'w', encoding="utf-8")
    for key, value in self.entity2id.items():
        f.write(key + '\t' + str(value) +'\n')
    f.close()

    f = open(relation_dic_path, 'w', encoding="utf-8")
    for key, value in self.relation2id.items():
        f.write(key + '\t' + str(value) +'\n')
    f.close()

    return entity2id, relation2id


class Dataset(data.Dataset):
    """Dataset implementation for handling FB15K and FB15K-237."""

    def __init__(self, data_path: str, entity2id: Mapping, relation2id: Mapping):
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.data

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        head, relation, tail = self.data[index]
        head_id = self._to_idx(head, self.entity2id)
        relation_id = self._to_idx(relation, self.relation2id)
        tail_id = self._to_idx(tail, self.entity2id)
        return head_id, relation_id, tail_id

    @staticmethod
    def _to_idx(key: str, mapping: Mapping) -> int:
        try:
            return mapping[key]
        except KeyError:
            return len(mapping)
