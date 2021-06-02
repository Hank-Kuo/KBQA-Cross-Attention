import torch
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_

class Net(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, 
                relation_dim, num_entities, pretrained_embeddings, 
                device, entdrop, reldrop, scoredrop, l3_reg,
                label_smoothing, w_matrix, bn_list):
        super(Net, self).__init__()
        self.device = device
        self.bn_list = bn_list
        self.model = model
        self.label_smoothing = label_smoothing
        self.l3_reg = l3_reg
        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim # * multiplier
        self.n_layers = 1
        self.bidirectional = True
        self.num_entities = num_entities

        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=False)
        self.loss = torch.nn.BCELoss(reduction='sum')

        self.mid1 = 256
        self.mid2 = 256

        self.lin1 = nn.Linear(hidden_dim * 2, self.mid1, bias=False)
        self.lin2 = nn.Linear(self.mid1, self.mid2, bias=False)

        self.hidden2rel = nn.Linear(self.mid2, self.relation_dim)
        
        self.bn0 = torch.nn.BatchNorm1d(self.embedding.weight.size(1))
        self.bn2 = torch.nn.BatchNorm1d(self.embedding.weight.size(1))

        self.GRU = nn.LSTM(embedding_dim, self.hidden_dim, self.n_layers, 
                            bidirectional=self.bidirectional, batch_first=True) 

    def applyNonLinear(self, outputs):
        outputs = self.lin1(outputs)
        outputs = F.relu(outputs)
        outputs = self.lin2(outputs)
        outputs = F.relu(outputs)
        outputs = self.hidden2rel(outputs)
        return outputs
    
    def SimplE(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        s = head * relation
        s_head, s_tail = torch.chunk(s, 2, dim=1)
        s = torch.cat([s_tail, s_head], dim=1)
        s = self.bn2(s)
        s = self.score_dropout(s)
        s = torch.mm(s, self.embedding.weight.transpose(1,0))
        s = 0.5 * s
        pred = torch.sigmoid(s)
        return pred

    def ComplEx(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        score = self.bn2(score)
        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        pred = torch.sigmoid(score)
        return pred

    
    def forward(self, sentence, p_head, p_tail, question_len):
        embeds = self.word_embeddings(sentence)
        packed_output = pack_padded_sequence(embeds, question_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs, outputs_length = pad_packed_sequence(outputs, batch_first=True)
        outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        rel_embedding = self.applyNonLinear(outputs)
        p_head = self.embedding(p_head)
        pred = self.getScores(p_head, rel_embedding)
        actual = p_tail
        if self.label_smoothing:
            actual = ((1.0-self.label_smoothing)*actual) + (1.0/actual.size(1)) 
        loss = self.loss(pred, actual)
        if self.l3_reg:
            norm = torch.norm(self.embedding.weight, p=3, dim=-1)
            loss = loss + self.l3_reg * torch.sum(norm)
        return loss
        
    def get_relation_embedding(self, head, sentence, sent_len):
        embeds = self.word_embeddings(sentence.unsqueeze(0))
        packed_output = pack_padded_sequence(embeds, sent_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        rel_embedding = self.applyNonLinear(outputs)
        return rel_embedding

    def get_score_ranked(self, head, sentence, sent_len):
        embeds = self.word_embeddings(sentence.unsqueeze(0))
        packed_output = pack_padded_sequence(embeds, sent_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        rel_embedding = self.applyNonLinear(outputs)

        head = self.embedding(head).unsqueeze(0)
        score = self.getScores(head, rel_embedding)
        
        top2 = torch.topk(score, k=2, largest=True, sorted=True)
        return top2
        
