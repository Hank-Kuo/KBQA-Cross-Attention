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
                device, entdrop, reldrop, scoredrop, bn_list):
        super(Net, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.bn_list = bn_list
        multiplier = 2 
        self.n_layers = 1
        self.bidirectional = True
        self.num_entities = num_entities

        #self.ent_dropout = torch.nn.Dropout(entdrop)
        #self.rel_dropout = torch.nn.Dropout(reldrop)
        #self.score_dropout = torch.nn.Dropout(scoredrop)

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.GRU = nn.LSTM(embedding_dim, self.hidden_dim, self.n_layers, 
                            bidirectional=self.bidirectional, batch_first=True) 
        # KB embedding                             
        self.KB_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=True)
        self.loss = torch.nn.BCELoss(reduction='sum')

    def ComplEx(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        
        # head = self.ent_dropout(head)
        # relation = self.rel_dropout(relation)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.KB_embedding.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        
        # score = self.score_dropout(score)
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
        rel_embedding = outputs

        p_head = self.KB_embedding(p_head)
        pred = self.ComplEx(p_head, rel_embedding)
        actual = p_tail
        loss = self.loss(pred, actual)
        
        return loss
        
    def get_relation_embedding(self, head, sentence, sent_len):
        embeds = self.word_embeddings(sentence.unsqueeze(0))
        packed_output = pack_padded_sequence(embeds, sent_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        rel_embedding = outputs
        return rel_embedding

    def get_score_ranked(self, head, sentence, sent_len):
        embeds = self.word_embeddings(sentence.unsqueeze(0))
        packed_output = pack_padded_sequence(embeds, sent_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        rel_embedding = outputs

        head = self.KB_embedding(head).unsqueeze(0)
        score = self.ComplEx(head, rel_embedding)
        
        top2 = torch.topk(score, k=2, largest=True, sorted=True)
        return top2
        
