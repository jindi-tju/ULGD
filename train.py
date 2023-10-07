import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
import networkx as nx
import random
import math, os
from collections import defaultdict
from models import DGI, LogReg, DGI_gat
from utils import process
from attacker.attacker import Attacker
from estimator.estimator import mi_loss, mi_loss_gat, mi_loss_neg, mi_loss_attention
from sklearn.cluster import KMeans
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Process some integers.')
 
parser.add_argument('--dataset', type=str, default='polblogs', help='polblogs')  
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--tau', type=float, default=0.01)
parser.add_argument('--critic', type=str, default="bilinear")
parser.add_argument('--hinge', type=bool, default=True)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--save-model', type=bool, default=True)
parser.add_argument('--show-task', type=bool, default=True)
parser.add_argument('--show-attack', type=bool, default=True)

args = parser.parse_args()

dataset = args.dataset

print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

make_adv = True
attack_rate = args.alpha

# training params
batch_size = 1
nb_epochs =500  
patience = 30
gat_epoch = 1  
attack_iters = 1 
lr = 0.005
lr_gat = 0.0015
l2_coef = 0
drop_prob = 0.0
k_1 = 1  
k_2 = 0.1
Value = 0.2
step_size_init = 10 
stepsize_x = 5e-5
hid_units = args.dim
sparse = True
if dataset == 'polblogs':
    attack_mode = 'A'
else:
    attack_mode = 'A'
nonlinearity = 'prelu'  

if dataset == 'polblogs':
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data_polblogs(dataset)
else:
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
adj1 = np.loadtxt(fname="blog.txt", delimiter=' ', encoding='utf-8')
adj_ori = adj.todense()
adj2 = torch.tensor(adj1).cuda()
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]
nb_edges = int(adj.sum() / 2)
n_flips = int(nb_edges * attack_rate)
A = adj.copy()
A_self = A.copy()
features, _ = process.preprocess_features(features, dataset=dataset)

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
A_self = A_self + sp.eye(adj.shape[0])
if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    sp_A = process.sparse_mx_to_torch_sparse_tensor(A)
    sp_A_self = process.sparse_mx_to_torch_sparse_tensor(A_self)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()
features = torch.FloatTensor(features[np.newaxis])
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

if torch.cuda.is_available():
    print('Using CUDA')
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
        sp_A = sp_A.cuda()
        sp_A_self = sp_A_self.cuda()
    else:
        adj = adj.cuda()
        A = A.cuda()
        A_self = A_self.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

sp_adj = sp_adj.to_dense()
sp_adj_ori = sp_adj.clone()
features_ori = features.clone()
sp_A = sp_A.to_dense()
sp_A_self = sp_A_self.to_dense()
sp_adj_gat_last = sp_A_self.clone()
encoder = DGI(ft_size, hid_units, nonlinearity, critic=args.critic)
encoder_gat = DGI_gat(ft_size, hid_units, nonlinearity, critic=args.critic)
atm = Attacker(encoder, features, nb_nodes, attack_mode=attack_mode,
               show_attack=args.show_attack, gpu=torch.cuda.is_available())

optimiser = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=l2_coef)
optimiser_gat = torch.optim.Adam(encoder_gat.parameters(), lr=lr_gat, weight_decay=l2_coef)
if torch.cuda.is_available():
    encoder.cuda()
    encoder_gat.cuda()
    atm.cuda()

b_xent = nn.BCEWithLogitsLoss()
L2_xent = nn.MSELoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

drop = 0.8
epochs_drop = 20

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)


def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


def task(embeds):
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    if torch.cuda.is_available():
        log.cuda()

    for _ in range(500):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)
        loss.backward()
        opt.step()
    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    F1 = f1_score(preds.cpu(), test_lbls.cpu(), average='macro')
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    test_nmi = normalized_mutual_info_score(preds.cpu(), test_lbls.cpu(),
                                            average_method='arithmetic')
    test_ari = adjusted_rand_score(preds.cpu(), test_lbls.cpu(), )
    return acc.detach().cpu().numpy(), test_nmi


sp_B = sp_A_self.cuda()
sp_A_self = sp_A_self + sp_A_self - torch.ones(nb_nodes, nb_nodes).cuda()
original_acc_nat = 0
original_NMI = 0
for epoch in range(nb_epochs):
    encoder.train()
    optimiser.zero_grad() 

    if make_adv:
        step_size = step_size_init
        step_size_x = stepsize_x
        adv = atm(sp_adj, sp_A, None, n_flips, b_xent=b_xent, step_size=step_size,
                  eps_x=args.epsilon, step_size_x=step_size_x,
                  iterations=attack_iters, should_normalize=True, random_restarts=False,
                  make_adv=True) 
        if attack_mode == 'A':
            sp_adj = adv[0] 
            sp_A_gat = adv[1] 
            sp_A_gat = torch.where(sp_adj < 0.001, torch.zeros(nb_nodes, nb_nodes).cuda(),torch.ones(nb_nodes, nb_nodes).cuda())
        elif attack_mode == 'X':
            features = adv
        elif attack_mode == 'both':
            sp_adj = adv[0]
            features = adv[1]
    loss = mi_loss(encoder, sp_adj, features, nb_nodes, b_xent, batch_size, sparse) 
    print(loss)
    if cnt_wait == patience:
        print('Early stopping!')
        break
    sp_C = sp_A_gat - torch.eye(nb_nodes, nb_nodes).cuda()
    sp_adj_gat = sp_A_gat.detach()
    sp_D = torch.where(sp_C < 0.5, torch.zeros(nb_nodes, nb_nodes).cuda(),torch.ones(nb_nodes, nb_nodes).cuda())  # 攻击后的邻接矩阵
    k = 0
    if epoch % 1 == 0:
        loss_original = mi_loss_gat(encoder_gat, sp_adj_gat_last, features, nb_nodes, b_xent, batch_size, sparse)
        loss_now = mi_loss_gat(encoder_gat, sp_adj_gat, features, nb_nodes, b_xent, batch_size, sparse)
        vulnerabilities_value = loss_now.detach() - loss_original
        for x in range(gat_epoch):
            k += 1
            encoder_gat.train()
            optimiser_gat.zero_grad() 
            loss_gat_comparative = mi_loss_gat(encoder_gat, sp_adj_gat, features, nb_nodes, b_xent, batch_size, sparse)
            attention_adj = encoder_gat.attention_adj(features_ori, sp_adj_gat, sparse)
            e_softmax = encoder_gat.e_softmax(features_ori, sp_adj_gat, sparse)
            loss_gat_attention = mi_loss_attention(sp_A_self, attention_adj, nb_nodes, L2_xent, batch_size, sparse)
            print(vulnerabilities_value)
            if vulnerabilities_value >= Value:
                loss_gat = k_1 * loss_gat_comparative + k_2 * loss_gat_attention
            else:
                loss_gat = loss_gat_comparative
            print(loss_gat_comparative)
            print('best_Acc:{:.4f}'.format(original_acc_nat))
            loss_gat.backward()
            optimiser_gat.step()
            embeds, _ = encoder_gat.embed(features_ori, adj2, sparse, None)
            acc_nat = task(embeds)[0]
            NMI = task(embeds)[1]
            clean_embedding, _ = encoder_gat.embed(features_ori, sp_B, sparse, None)
            clean_embedding_acc = task(clean_embedding)[0]
            print('Epoch:{} Step_size: {:.4f} Loss:{:.4f} adv_Acc:{:.4f} clean_Acc:{:.4f}'.format(
                epoch, k, loss_gat.detach().cpu().numpy(), acc_nat, clean_embedding_acc))
            if acc_nat > original_acc_nat and epoch >= 0:
                original_acc_nat = acc_nat
            if NMI > original_NMI and epoch >= 0:
                original_NMI = NMI
        print('best_Acc:{:.4f}'.format(original_acc_nat))
        print('best_NMI:{:.4f}'.format(original_NMI))
        print('vulnerabilities_value:{:.4f}'.format(vulnerabilities_value))
    loss.backward()  
    optimiser.step() 
