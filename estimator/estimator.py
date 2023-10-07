import torch
import numpy as np

def mi_loss(encoder, sp_adj, features, nb_nodes, b_xent, batch_size=1, sparse=True, encoder_cpu=False):
    # if encoder_cpu:
    #     sp_adj = sp_adj.cpu()
    #     features = features.cpu()
    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]#(1,2078,1433)

    lbl_1 = torch.ones(batch_size, nb_nodes) #1*2708
    lbl_2 = torch.zeros(batch_size, nb_nodes)#1*2708
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available(): #1*5416
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()

    #logits = encoder(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)
    logits = encoder(features, shuf_fts, sp_adj, sparse, None, None, None)#(1*5416)
    loss = b_xent(logits, lbl)
    # return loss.view(1, -1)
    return loss


def mi_loss_gat(encoder, sp_adj, features, nb_nodes, b_xent, batch_size=1, sparse=True, encoder_cpu=False):
    # if encoder_cpu:
    #     sp_adj = sp_adj.cpu()
    #     features = features.cpu()
    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]#(1,2078,1433)

    lbl_1 = torch.ones(batch_size, nb_nodes) #1*2708
    lbl_2 = torch.zeros(batch_size, nb_nodes)#1*2708
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available(): #1*5416
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()

    #logits = encoder(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)
    logits = encoder(features, shuf_fts, sp_adj, sparse, None, None, None)#(1*5416)
    loss = b_xent(logits, lbl)
    # return loss.view(1, -1)
    return loss

def mi_loss_attention(sp_A_self, attention_adj, nb_nodes, L2_xent, batch_size=1, sparse=True, encoder_cpu=False):
    add =torch.mul(attention_adj,sp_A_self)
    value = add.sum()/nb_nodes
    lbl = torch.ones(1)
    lbl[0] = 1
    if torch.cuda.is_available():
        lbl = lbl.cuda()
    if  value >= 1:
        loss = L2_xent(value, value)
    else:
        loss = L2_xent(value, lbl)
    return loss
def mi_loss_neg(encoder, sp_adj, sp_adj_ori, features, nb_nodes, b_xent, batch_size=1, sparse=True):
    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2, lbl_2), 1)

    if torch.cuda.is_available():
        shuf_fts = shuf_fts.cuda()
        lbl = lbl.cuda()

    # logits = encoder(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)
    logits = encoder(features, shuf_fts, sp_adj, sparse, None, None, None)
    logits_ori_neg = encoder(features, shuf_fts, sp_adj_ori, sparse, None, None, None)[:, -lbl_2.shape[-1]:]

    logits = torch.cat((logits,logits_ori_neg), 1)
    loss = b_xent(logits, lbl)
    # return loss.view(1, -1)
    return loss

def mi_loss_ind(nodes, dgi, b_xent, batch_size=1, sparse=True):

    lbl_1 = torch.ones(1, batch_size)
    lbl_2 = torch.zeros(1, batch_size)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    if torch.cuda.is_available():
        lbl = lbl.cuda()

    logits = dgi(nodes, None, None, None)
    loss = b_xent(logits, lbl)
    # return loss.view(1, -1)
    return loss