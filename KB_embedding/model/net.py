import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, entity_count, relation_count, device, norm=1, dim=100):
        super(Net, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.dim = dim

        multiplier = 2 

        self.entities_emb = nn.Embedding(num_embeddings=self.entity_count + 1,
                                    embedding_dim=self.dim * multiplier,
                                    padding_idx=self.entity_count)
        self.relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
                                     embedding_dim=self.dim * multiplier,
                                     padding_idx=self.relation_count)
        self.head_dropout = torch.nn.Dropout(0.3)
        self.relation_dropout = torch.nn.Dropout(0.4)
        self.score_dropout = torch.nn.Dropout(0.5)
        self.head_batchNormal = torch.nn.BatchNorm1d(multiplier)
        self.relation_batchNormal = torch.nn.BatchNorm1d(multiplier)
        self.score_batchNormal = torch.nn.BatchNorm1d(multiplier)                              

        self.l3_reg = 0     
        self.criterion = torch.nn.BCELoss()
        

    def forward(self, head, relation, tail):
        heads = self.entities_emb(head)
        relations = self.relations_emb(relation)
        pred = self.ComplEx(heads, relations)
        
        loss = self.loss(pred, tail)
        return loss

    def loss(self, pred, actual):
        loss = self.criterion(pred, actual)
        return loss
        

    def ComplEx(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        head = self.head_batchNormal(head)
        head = self.head_dropout(head)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        relation = self.relation_dropout(relation)
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.E.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        score = self.score_batchNormal(score)
        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)
        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0)) 
        pred = torch.sigmoid(score)
        return pred
                                                                                                         


def hit_at_k(predictions: torch.Tensor, ground_truth_idx: torch.Tensor, device: torch.device, k: int = 10) -> int:
    assert predictions.size(0) == ground_truth_idx.size(0)

    zero_tensor = torch.tensor([0], device=device)
    one_tensor = torch.tensor([1], device=device)
    _, indices = predictions.topk(k=k, largest=False)
    return torch.where(indices == ground_truth_idx, one_tensor, zero_tensor).sum().item()


def mrr(predictions: torch.Tensor, ground_truth_idx: torch.Tensor) -> float:
    assert predictions.size(0) == ground_truth_idx.size(0)

    indices = predictions.argsort()
    return (indices == ground_truth_idx).nonzero()[:, 1].float().add(1.0).sum().item()

metrics = {
    'hit_at_k': hit_at_k,
    'mrr': mrr
}