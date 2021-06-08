import logging
import json 
import os

import torch
from torch import nn
from torch.optim import optimizer
from typing import Tuple

_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
_EPOCH = "epoch"
_BEST_SCORE = "best_score"

class Params():
    def __init__(self, json_path):
        if os.path.exists(json_path):
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)
        else:
            with open(json_path, 'w') as f:
                print('create json file')

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__



def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)



def load_checkpoint(checkpoint_dir: str, model: nn.Module, optim: optimizer.Optimizer) -> Tuple[int, int, float]:
    """Loads training checkpoint.

    :param checkpoint_path: path to checkpoint
    :param model: model to update state
    :param optim: optimizer to  update state
    :return tuple of starting epoch id, starting step id, best checkpoint score
    """
    if not os.path.exists(checkpoint_dir):
        raise ("File doesn't exist {}".format(checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.tar') 
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[_MODEL_STATE_DICT])
    optim.load_state_dict(checkpoint[_OPTIMIZER_STATE_DICT])
    start_epoch_id = checkpoint[_EPOCH] + 1
    
    best_score = checkpoint[_BEST_SCORE]
    return start_epoch_id, best_score


def save_checkpoint(checkpoint_dir: str, model: nn.Module, optim: optimizer.Optimizer, epoch_id: int, best_score: float):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.tar') 
    
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optim.state_dict(),
        _EPOCH: epoch_id,
        _BEST_SCORE: best_score
    }, checkpoint_path)


def write_embedding_files(self, output_dir, model):
        model.eval()
        model_folder = f"../kg_embeddings/{self.model}/{self.dataset}" 
        data_folder = "../data/%s/" % self.dataset
        embedding_type = self.model
        if(not os.path.exists(model_folder)):
            os.makedirs(model_folder)
        R_numpy = model.R.weight.data.cpu().numpy()
        E_numpy = model.E.weight.data.cpu().numpy()
        bn_list = []
        for bn in [model.bn0, model.bn1, model.bn2]:
            bn_weight = bn.weight.data.cpu().numpy()
            bn_bias = bn.bias.data.cpu().numpy()
            bn_running_mean = bn.running_mean.data.cpu().numpy()
            bn_running_var = bn.running_var.data.cpu().numpy()
            bn_numpy = {}
            bn_numpy['weight'] = bn_weight
            bn_numpy['bias'] = bn_bias
            bn_numpy['running_mean'] = bn_running_mean
            bn_numpy['running_var'] = bn_running_var
            bn_list.append(bn_numpy)
            
        if embedding_type == 'TuckER':
            W_numpy = model.W.detach().cpu().numpy()
            
        np.save(model_folder +'/E.npy', E_numpy)
        np.save(model_folder +'/R.npy', R_numpy)
        for i, bn in enumerate(bn_list):
            np.save(model_folder + '/bn' + str(i) + '.npy', bn)

        if embedding_type == 'TuckER':
            np.save(model_folder +'/W.npy', W_numpy)

        f = open(data_folder + '/entities.dict', 'r')
        f2 = open(model_folder + '/entities.dict', 'w')
        ents = {}
        idx2ent = {}
        for line in f:
            line = line.rstrip().split('\t')
            name = line[0]
            id = int(line[1])
            ents[name] = id
            idx2ent[id] = name
            f2.write(str(id) + '\t' + name + '\n')
        f.close()
        f2.close()
        f = open(data_folder + '/relations.dict', 'r')
        f2 = open(model_folder + '/relations.dict', 'w')
        rels = {}
        idx2rel = {}
        for line in f:
            line = line.strip().split('\t')
            name = line[0]
            id = int(line[1])
            rels[name] = id
            idx2rel[id] = name
            f2.write(str(id) + '\t' + name + '\n')
        f.close()
        f2.close()