import argparse
import os
import logging
import numpy as np
from tqdm import tqdm

import model.net as net
import model.data_loader as data_loader
import utils
from evaluate import evaluate
from preprocess import process_KB, process_text_file, get_vocab, prepare_embeddings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as torch_data
from torch.optim.lr_scheduler import ExponentialLR

parser = argparse.ArgumentParser(description='KBQA with LSTM PyTorch')
parser.add_argument("--dataset_path", default="./data", help="Path to dataset.")
parser.add_argument("--model_dir", default="./experiments/base_model", help="Path to model checkpoint (by default train from scratch).")

def main():
    args = parser.parse_args()

    # torch setting
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # path setting
    path = args.dataset_path
    train_dataset_path = os.path.join(path, "train/train.json")
    valid_dataset_path = os.path.join(path, "valid/valid.json")
    test_dataset_path = os.path.join(path, "test/test.json")
    entity_dict_path = os.path.join(path, "KB/entities.json")
    relation_dict_path = os.path.join(path, "KB/relations.json")
    checkpoint_dir = os.path.join(args.model_dir, "checkpoint")
    params_path = os.path.join(args.model_dir, 'params.json')
    
    # params
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    params = utils.Params(params_path)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # KB
    logging.info("Loading KB datasets...")
    
    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    
    # question text
    train_data = process_text_file(train_dataset_path, split=False)
    valid_data = process_text_file(valid_dataset_path)
    word2ix, idx2word, max_len = get_vocab(train_data)
    
    # data loader
    logging.info("Loading QA datasets...")
    dataset = data_loader.MetaQADataset(data=train_data, word2ix=word2ix, relations=r, entities=e, entity2idx=entity2idx)
    data_generator = data_loader.MetaQADataLoader(dataset, batch_size=params.batch_size, shuffle=True)
    
    # net params
    model = net.Net(embedding_dim=params.embedding_dim, hidden_dim=params.hidden_dim, vocab_size=len(word2ix), 
                    num_entities = len(idx2entity), relation_dim=params.relation_dim, device=params.device, 
                    entdrop = params.entdrop, reldrop = params.reldrop, scoredrop = params.scoredrop,
                    pretrained_embeddings=embedding_matrix)
    model.to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    scheduler = ExponentialLR(optimizer, params.decay)
    optimizer.zero_grad()
    best_score = -float("inf")
    no_update = 0
    
    logging.info('----------------------------------------------')
    logging.info("Number of KB's Entity: {}, Number of KB's Relation : {}".format(len(e), len(r)))
    logging.info("PreTrain KB's Entity size: {}, PreTrain KB's Relation size: {}".format(entities.shape, relations.shape))
    logging.info("Number of Vocab: {}, Max length word: {}".format(len(word2ix), max_len))
    logging.info("Number of Train dataset: {}, Number of Valild dataset: {}".format(len(train_data), len(valid_data)))
    print(model)

    logging.info("Starting training...")
    for epoch_id in range(1, params.epochs):
        # train mod
        print("Epoch {}/{}".format(epoch_id, params.epochs))

        model.train()
        running_loss = 0
        with tqdm(total=len(data_generator)) as t:
            for i_batch, a in enumerate(data_generator):
                model.zero_grad()
                question = a[0].to(params.device)
                sent_len = a[1]
                positive_head = a[2].to(params.device)
                positive_tail = a[3].to(params.device)                    

                loss = model(sentence=question, p_head=positive_head, p_tail=positive_tail, question_len=sent_len)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                t.set_postfix(loss=running_loss/((i_batch+1)*params.batch_size))
                t.update()
            scheduler.step()
            
            # valid mod
            if epoch_id % params.valid_every == 0 :
                model.eval()
                eps = 0.0001
                answers, score = evaluate(model=model, data=valid_data, word2idx=word2ix, entity2idx= entity2idx, 
                                        device=params.device)
                logging.info("- Eval metrics: {}".format(score))
                if score > best_score + eps:
                    # update 
                    best_score = score
                    no_update = 0
                    logging.info("- Test score for best valid so far {}".format(best_score))
                    utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_id, best_score)
                elif (score < best_score + eps) and (no_update < params.patience):
                    no_update +=1
                    logging.info("- Validation accuracy decreases to {} from {}, {} more epoch to check".format(score, best_score, params.patience-no_update))
                elif no_update == params.patience:
                    # early stopping
                    logging.info("- Model has exceed patience. Saving best model and exiting")
                    utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_id, best_score)
                    exit()
                if epoch_id == params.epochs -1:
                    logging.info("- Final Epoch has reached. Stopping and saving model")
                    utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_id, best_score)
                    exit()

if __name__=='__main__':
    main()