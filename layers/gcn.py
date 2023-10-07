import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, act, sparse=False):
        seq_fts = self.fc(seq)
        #if sparse:
        out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0) #矩阵乘法
        #else:
        #    out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class GAT(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
#    def __init__(self, in_features, out_features, dropout = 0.5, alpha = 0.2, concat=True):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.05, concat=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.fc = nn.Linear(in_features, in_features, bias=False) 
        nn.init.xavier_uniform_(self.fc.weight.data) 
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))) 
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        #for m in self.modules(): 
        #    print(m)
        #    self.weights_init(m)

    #def weights_init(self, m):
    #    if isinstance(m, nn.Linear):
    #        torch.nn.init.xavier_uniform_(m.weight.data)
    #        if m.bias is not None:
    #            m.bias.data.fill_(0.0)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, seq, adj, sparse=False):
        Wh = torch.mm(torch.squeeze(seq, 0), self.W) 
        e = self._prepare_attentional_mechanism_input(Wh) 
        zero_vec = -9e15*torch.ones_like(e)
        e_softmax = F.softmax(e, dim=1)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime =  torch.unsqueeze(torch.matmul(attention, Wh), 0)
        k=F.elu(h_prime)
        return  F.elu(h_prime), attention, e_softmax #e_softmax
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class Attention(nn.Module):
    def __init__(self, in_dim, dropout):
        super(Attention, self).__init__()
        self.dropout = dropout
        self.a1 = nn.Linear(512, 1)
        self.a2 = nn.Linear(512, 1)
        self.fc = nn.Linear(in_dim, 512, bias=False)  
        nn.init.xavier_uniform_(self.fc.weight.data)  

    def forward(self, seq, adj, sparse=False):
        x = torch.squeeze(self.fc(seq), 0)
        f1 = self.a1(x)
        f2 = self.a2(x)
        e = F.leaky_relu(f1 + f2.t(), 0.2)
        if adj is None:
            score = e
        else:
            zero_vec = -9e15 * torch.ones_like(e)
            score = torch.where(adj > 0, e, zero_vec)
        score = F.softmax(score, dim=1)
        score = F.dropout(score, 0.5, training=self.training)
        h = torch.unsqueeze(torch.matmul(score, x),0)

        return h