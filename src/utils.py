from difflib import Differ
import torch
import numpy as np
import random
from copy import deepcopy


def diff_string(s1, s2):
    # return: index to convert s1 -> s2
    differ = Differ()
    diff = list(differ.compare(s1, s2))
    diff_idx = [0] * len(s1)
    count = -1
    add = -1
    for d in diff:
        mode = d[0]
        ch = d[2]
        if mode == "+":
            add += 1
        elif mode == "-":
            count += 1
            diff_idx[count] = diff_idx[count - 1]
            # count+=1
        else:
            count += 1
            diff_idx[count] = diff_idx[count - 1] + add + 1
            add = 0
    # assert ''.join([s2[d] for d in diff_idx if d>0]) == s1
    for i in range(count, len(s1)):
        diff_idx[i] = diff_idx[i - 1] + 1
    return diff_idx


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
