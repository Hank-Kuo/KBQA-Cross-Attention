import os
import argparse
from tqdm import tqdm

import model.net as net
import model.data_loader as data_loader
import utils
from evaluate import evaluate

import torch
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils import tensorboard

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default="../data/KB", help="Path to dataset.")
parser.add_argument("--model_dir", default="./experiments/base_model", help="Path to model checkpoint (by default train from scratch).")


def main():
    args = parser.parse_args()

    # torch setting
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # os setting
    path = args.dataset_path
    train_path = os.path.join(path, "train.txt")
    validation_path = os.path.join(path, "valid.txt")
    test_path = os.path.join(path, "test.txt")
    params_path = os.path.join(args.model_dir, 'params.json')
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoint')
    
    # params
    params = utils.Params(params_path)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # dataset
    print("Process entities and relations in KB")
    train_data = data_loader.load_data(train_path, reverse=True)
    test_data = data_loader.load_data(test_path, reverse=True)
    valid_data = data_loader.load_data(validation_path, reverse=True)
    entity2id, relation2id = data_loader.create_mappings(train_path, valid_data, test_path, path)

    # dataset
    print("Process dataset...")
    train_set = data_loader.Dataset(train_data, entity2id, relation2id)
    train_generator = torch_data.DataLoader(train_set, batch_size=params.batch_size)
    validation_set = data_loader.Dataset(valid_data, entity2id, relation2id)
    validation_generator = torch_data.DataLoader(validation_set, batch_size=params.valid_batch_size)
    test_set = data_loader.Dataset(test_data, entity2id, relation2id)
    test_generator = torch_data.DataLoader(test_set, batch_size=1)

    # model
    model = net.Net(entity_count=len(entity2id), relation_count=len(relation2id), dim=params.embedding_dim,
                                    device=params.device, head_dropout=params.head_dropout, relation_dropout=params.relation_dropout
                                    score_dropout=params.score_dropout)
    model = model.to(params.device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    step = 0
    best_score = 0.0

    print("KB Dataset: entity: {} relation: {}".format(len(entity2id), len(relation2id)))
    print("Train Dataset: triples: {}".format(len(train_set)))
    print("Validation Dataset: triples: {}".format(len(validation_set)))
    print("Test Dataset: triples: {}".format(len(test_set)))
    print(model)

    # Train
    for epoch_id in range(1, params.epochs + 1):
        print("Epoch {}/{}".format(epoch_id, params.epochs))
        
        running_loss = 0
        running_batch = 0
        model.train()

        with tqdm(total=len(train_generator)) as t:
            for local_heads, local_relations, local_tails in train_generator:
                local_heads, local_relations, local_tails = (local_heads.to(params.device), local_relations.to(params.device),
                                                            local_tails.to(params.device))

                # scheduler = ExponentialLR(opt, params.decay_rate)
                optimizer.zero_grad()
                loss = model(local_heads, local_relations, local_tails)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_batch += local_heads.size()[0]
                
                t.set_postfix(loss = running_loss / running_batch * 100)
                t.update()

            # validation
            if epoch_id % params.validation_freq == 0:
                model.eval()
                _, _, hits_at_10, _ = evaluate(model=model, data_generator=validation_generator,
                                        entities_count=len(entity2id),
                                        device=params.device, summary_writer=summary_writer,
                                        epoch_id=epoch_id, metric_suffix="val")
                score = hits_at_10
                if score > best_score:
                    best_score = score
                    utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_id, step, best_score)
            



if __name__ == '__main__':
    main()
