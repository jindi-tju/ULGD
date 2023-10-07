import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np

from layers import GCN, AvgReadout, Discriminator, Discriminator2, Discriminator_jsd, MeanAggregator, MeanAggregator_ml, Encoder, GAT, Attention
from estimator.estimator import mi_loss
from attacker.attacker import Attacker

class DGI(nn.Module): 
    def __init__(self, n_in, n_h, activation, critic="bilinear", dataset=None, attack_model=True):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h, critic=critic, dataset=dataset, attack_model=attack_model)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2): 
        h_1 = self.gcn(seq1, adj, sparse)   

        c = self.read(h_1, msk)   
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse) 

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)    

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk, grad=False):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        if grad:
            return h_1, c
        else:
            return h_1.detach(), c.detach()


class DGI_local(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI_local, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        #self.gcn = GAT(n_in, n_h)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc1 = Discriminator(n_h)
        self.disc2 = Discriminator2(n_in, n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)  

        c = self.read(h_1, msk)  
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret1 = self.disc1(c, h_1, h_2, samp_bias1, samp_bias2) 
        return ret1

    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class DGI_ind(nn.Module):
    def __init__(self, features, adj_lists, ft_size, n_h, activation, num_sample=[10, 10], skip_connection=False, gcn=True):
        super(DGI_ind, self).__init__()
        self.features = features
        self.skip_connection = skip_connection
        self.agg1 = MeanAggregator(features, cuda=torch.cuda.is_available(), gcn=gcn, name='l1')
        self.enc1 = Encoder(features, ft_size, n_h, adj_lists, self.agg1, num_sample=num_sample[0], gcn=gcn,
                            cuda=torch.cuda.is_available(), activation=activation, skip_connection=skip_connection, name='l2')
        self.agg2 = MeanAggregator(lambda nodes: self.enc1(nodes), cuda=torch.cuda.is_available(), gcn=gcn, name='l3')
        self.enc2 = Encoder(lambda nodes: self.enc1(nodes), self.enc1.embed_dim, n_h, adj_lists, self.agg2, num_sample=num_sample[1],
                            base_model=self.enc1, gcn=gcn, cuda=torch.cuda.is_available(),
                            activation=activation, skip_connection=skip_connection, name='l4')
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()

        if skip_connection:
            self.disc = Discriminator(2*n_h)
        else:
            self.disc = Discriminator(n_h)

    def forward(self, nodes, msk, samp_bias1, samp_bias2, make_adv=True):
        h_1, h_2 = self.enc2(nodes)

        h_1 = h_1.transpose(0,1).unsqueeze(0)
        h_2 = h_2.transpose(0,1).unsqueeze(0)
        c = self.read(h_1, msk)  
        c = self.sigm(c)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2) 

        return ret

    # Detach the return variables
    def embed(self, nodes, msk):
        h_1, _ = self.enc2(nodes)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class DGI_ind_ml(nn.Module):
    def __init__(self, features, adj_lists, ft_size, n_h, activation, num_sample=[10, 10, 25], skip_connection=False, gcn=True):
        super(DGI_ind_ml, self).__init__()
        self.features = features
        self.skip_connection = skip_connection
        self.agg1 = MeanAggregator_ml(features, cuda=torch.cuda.is_available(), gcn=gcn, name='l1')
        self.enc1 = Encoder(features, ft_size, n_h, adj_lists, self.agg1, num_sample=num_sample[0], gcn=gcn,
                            cuda=torch.cuda.is_available(), activation=activation, skip_connection=skip_connection, name='l2')
        self.agg2 = MeanAggregator_ml(lambda nodes: self.enc1(nodes), cuda=torch.cuda.is_available(), gcn=gcn, name='l3')
        self.enc2 = Encoder(lambda nodes: self.enc1(nodes), self.enc1.embed_dim, n_h, adj_lists, self.agg2, num_sample=num_sample[1],
                            base_model=self.enc1, gcn=gcn, cuda=torch.cuda.is_available(),
                            activation=activation, skip_connection=skip_connection, name='l4')
        self.agg3 = MeanAggregator_ml(lambda nodes: self.enc2(nodes), cuda=torch.cuda.is_available(), gcn=gcn, name='l5')
        self.enc3 = Encoder(lambda nodes: self.enc2(nodes), self.enc2.embed_dim, n_h, adj_lists, self.agg3, num_sample=num_sample[2],
                            base_model=self.enc2, gcn=gcn, cuda=torch.cuda.is_available(),
                            activation=activation, skip_connection=skip_connection, name='l6')
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()

        if skip_connection:
            self.disc = Discriminator(2*n_h)
        else:
            self.disc = Discriminator(n_h)

    def forward(self, nodes, msk, samp_bias1, samp_bias2, make_adv=True):
        h_1, h_2 = self.enc3(nodes)

        h_1 = h_1.transpose(0,1).unsqueeze(0)
        h_2 = h_2.transpose(0,1).unsqueeze(0)

        c = self.read(h_1, msk)  
        c = self.sigm(c)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)  
        return ret

    def embed(self, nodes, msk):
        h_1, _ = self.enc2(nodes)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class DGI_gat(nn.Module): 
    def __init__(self, n_in, n_h, activation, critic="bilinear", dataset=None, attack_model=True):
        super(DGI_gat, self).__init__()
        self.gcn = GAT(n_in, n_h)  
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h, critic=critic, dataset=dataset, attack_model=attack_model)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2): 
        h_1 = self.gcn(seq1, adj, sparse)[0]   
        c = self.read(h_1, msk)  
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)[0]  

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)    

        return ret

    def attention_adj(self, seq, adj, sparse, grad=True):
        attention = self.gcn(seq, adj, sparse)[1]
        return attention

    def e_softmax(self, seq, adj, sparse, grad=True):
        e_softmax = self.gcn(seq, adj, sparse)[2]
        return e_softmax

    def embed(self, seq, adj, sparse, msk, grad=False):
        h_1 = self.gcn(seq, adj, sparse)[0]
        c = self.read(h_1, msk)

        if grad:
            return h_1, c
        else:
            return h_1.detach(), c.detach()