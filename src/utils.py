from difflib import Differ
import torch
import numpy as np
import random
from copy import deepcopy


def calc_score(pred, label):
    pred = pred != 0
    label = label != 0
    tp = (pred * label).sum().to(torch.float)
    fp = (pred * ~label).sum().to(torch.float)
    # tn = (~pred * ~label).sum()
    fn = (~pred * label).sum().to(torch.float)

    if tp == 0:
        recall = torch.tensor(0).to(pred).to(torch.float)
        precision = torch.tensor(0).to(pred).to(torch.float)
    else:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

    if recall == 0 and precision == 0:
        f1 = torch.tensor(0).to(pred).to(torch.float)
    else:
        f1 = 2 * recall * precision / (recall + precision)

    return precision, recall, f1

# TODO
def process_layout(G):
    # suport only DAG
    pos = np.zeros(G.number_of_nodes,2)
    opes = {}
    for i,(node,attrib) in enumerate(G.nodes.items()):
        opes[node] = attrib
    
    # sort opes
    sorted_opes = []
    tmp_opes = dict([(o,[]) for o in opes])
    for n1,n2 in G.edges:
        if n1 in opes and n2 in opes:
            tmp_opes[n1] = n2

    fin = list(tmp_opes.keys())
    for n,child in  tmp_opes.items():
        for c in child:
            pass
