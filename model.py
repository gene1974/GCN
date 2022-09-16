import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GCNLayer(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.weights = nn.Parameter(torch.DoubleTensor(in_feature, out_feature))
        self.bias = nn.Parameter(torch.DoubleTensor(out_feature))
        self._init_weights()

    def _init_weights(self):
        std = np.sqrt(self.bias.shape[0])
        self.weights.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
    
    def forward(self, h, mask = None):
        '''
        h: input vector, (n_batch, n_node, in_feature)
        mask: A_ij, (n_batch, n_node, n_node)
        '''
        d = torch.sum(mask, dim = -1, keepdim = True) + 1e-7 # (n_batch, n_node)
        h = torch.matmul(h, self.weights) # (n_batch, n_node, out_feature)
        h = torch.matmul(mask, h) # (n_batch, n_node, out_feature)
        h = F.relu(h / d + self.bias) # (n_batch, n_node, out_feature)
        return h

class GCN(nn.Module):
    def __init__(self, n_layer, emb_dim):
        super().__init__()
        self.n_layer = n_layer
        self.gcn_layers = nn.ModuleList([GCNLayer(emb_dim, emb_dim) for _ in range(n_layer)])
        
    def forward(self, tokens, dependency, subject, object):
        '''
        subject: (n_batch, 2)
        '''
        h = tokens # (n_batch, n_node, out_feature)
        for gcn in self.gcn_layers:
            h = gcn(h, dependency)
        
        return h

class GCNModel(nn.Module):
    def __init__(self, args, word_emb):
        super().__init__()
        self.args = args
        self.word_embedding = nn.Embedding.from_pretrained(word_emb)
        n_feature, n_hidden = word_emb.shape[1], 300
        if args.model == 'cgcn':
            self.lstm = nn.LSTM(n_feature, n_feature // 2, 2, batch_first = True, bidirectional = True)
        self.gcn = GCN(args.nlayer, n_feature)
        self.ffnn = nn.Sequential(
            nn.Linear(n_feature * 3, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )
        self.fc = nn.Linear(n_hidden, args.n_relation)
        
    def forward(self, tokens, dependency, subject, object):
        token_emb = self.word_embedding(tokens) # (n_batch, n_node, emb_dim)
        if self.args.model == 'cgcn':
            sen_len = torch.sum(tokens != 0, dim = 1, dtype = int).to('cpu') # (batch_size)
            pack_seq = pack_padded_sequence(token_emb, sen_len, batch_first = True, enforce_sorted = False)
            lstm_out, _ = self.lstm(pack_seq)
            token_rep, _ = pad_packed_sequence(lstm_out, batch_first = True) # (batch_size, seq_len, hidden_size)
            token_pad = torch.zeros((token_emb.shape[0], token_emb.shape[1] - token_rep.shape[1], token_emb.shape[2]), device = 'cuda')
            token_rep = torch.cat((token_rep, token_pad), dim = 1)
        else:
            token_rep = token_emb
        token_rep = F.dropout(token_rep, 0.2)
        
        h = self.gcn(token_rep, dependency, subject, object) # (16, 96, 100)
        h = F.dropout(h, 0.2)
        rep = []
        for i in range(h.shape[0]):
            hsent = torch.max(h[i], dim = -2).values
            hs = torch.max(h[i, subject[i, 0]: subject[i, 1] + 1], dim = -2).values
            ho = torch.max(h[i, object[i, 0]: object[i, 1] + 1], dim = -2).values
            rep.append(torch.cat((hs, hsent, ho), dim = -1))
        rep = torch.stack(rep, dim = 0) # (n_batch, n_feature * 3)
        
        rep = self.ffnn(rep)
        output = self.fc(rep) # (n_batch, n_relation)
        return output