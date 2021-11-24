import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib.colors import LogNorm
from argparse import ArgumentParser
from pathlib import Path
from relation_data import RelationData
from seqeval.metrics.sequence_labeling import classification_report

# from seqeval.metrics.sequence_labeling import get_entities
from seqeval.scheme import Tokens
from seqeval.scheme import IOB2


def get_entities(seq):
    return [(e.tag, e.start, e.end) for e in Tokens(seq, IOB2).entities]


# Confusion Matrix
def _data2seq(gold, pred):
    pseq = []
    gseq = []

    def rellist(data):
        lst = []
        for rel in data["relation"].values():
            arg1 = data["entity"][rel["arg1"]]
            arg2 = data["entity"][rel["arg2"]]
            d = (
                rel["label"],
                # arg1["entity"].replace(" ", ""),
                arg1["start"],
                arg1["end"],
                arg1["label"],
                # arg2["entity"].replace(" ", ""),
                arg2["start"],
                arg2["end"],
                arg2["label"],
            )
            lst.append(d)
        return lst

    glst = rellist(gold)
    plst = rellist(pred)
    gcls = [l[0] for l in glst]
    pcls = [l[0] for l in plst]
    glst = [l[1:] for l in glst]
    plst = [l[1:] for l in plst]

    def all_match(a, b):
        return all([x == y for x, y in zip(a, b)])

    for p, c in zip(plst, pcls):
        flag = False
        for x, gc in zip(glst, gcls):
            if all_match(p, x):
                pseq.append(c)
                gseq.append(gc)
                flag = True
                break
        if not flag:
            pseq.append(c)
            gseq.append("None")
    for g, c in zip(glst, gcls):
        flag = False
        for x, pc in zip(plst, pcls):
            if all_match(g, x):
                if pc != c:
                    pseq.append(pc)
                    gseq.append(c)
                flag = True
                break
        if not flag:
            pseq.append("None")
            gseq.append(c)
    tn = len(pred["entity"]) ** 2 - len(plst)
    gseq.extend(["None"] * (tn))
    pseq.extend(["None"] * (tn))
    return gseq, pseq


def get_confusion_matrix(gold_data, pred_data, labels=None):
    pseqs = []
    gseqs = []
    for key in gold_data.keys():
        gseq, pseq = _data2seq(gold_data[key], pred_data[key]) if key in pred_data.keys() else ([], [])
        gseqs.extend(gseq)
        pseqs.extend(pseq)
    conf = confusion_matrix(gseqs, pseqs, labels=labels)
    return conf


def draw_confusion_matrix(conf, labels=None, figsize=(9.6, 7.2), label_fontsize=13, tick_fontsize=10):
    df = pd.DataFrame(data=conf, index=labels, columns=labels)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    sns.heatmap(
        df,
        square=True,
        cbar=True,
        annot=True,
        fmt="",
        cmap="jet",
        norm=LogNorm(),
        ax=ax,
    )
    ax.tick_params(axis="y", labelrotation=0, labelsize=tick_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.set_xlabel("Pred", fontsize=label_fontsize)
    ax.set_ylabel("Gold", fontsize=label_fontsize)
    fig.tight_layout()
    return fig


def confusion_matrix2metrics(conf, negative_indices=[], return_count=False):
    tp = fn = fp = 0
    tn = 0
    for i in range(len(conf)):
        if i in negative_indices:
            tn += conf[i][i]
        else:
            tp += conf[i][i]
        for j in range(len(conf)):
            if i == j:
                continue
            if j in negative_indices:
                fn += conf[i][j]
            elif i in negative_indices:
                fp += conf[i][j]
            else:
                fp += conf[i][j]
                fn += conf[i][j]
    if return_count:
        return tp, fp, fn, tn
    precision = tp / (tp + fp + 1e-20)
    recall = tp / (tp + fn + 1e-20)
    f_measure = 2 * precision * recall / (precision + recall + 1e-20)
    return precision, recall, f_measure


def bio2conf(gold, pred, sep="\t", strict=False, return_labels=False):
    gold = list(map(lambda x: [y.split(sep) for y in x.split("\n")], Path(gold).read_text().split("\n\n")))
    pred = list(map(lambda x: [y.split(sep) for y in x.split("\n")], Path(pred).read_text().split("\n\n")))
    gold = [x for x in gold if x[0][0]]
    pred = [x for x in pred if x[0][0]]
    gc = [[y[1] for y in x] for x in gold if x[0][0]]
    pc = [[y[1] for y in x] for x in pred if x[0][0]]
    print(classification_report(gc, pc, mode="strict" if strict else None, digits=4))

    gseq = []
    pseq = []

    def strict_match(a, b):
        return all([x == y for x, y in zip(a, b)])

    def soft_match(a, b):
        return any([x == y for x, y in zip(a, b)])

    match = strict_match if strict else soft_match
    tp = fp = fn = 0
    lst = []
    for g, p in zip(gold, pred):
        gws = [w[0] for w in g]
        pws = [w[0] for w in p]
        assert all([x == y for x, y in zip(gws, pws)])
        gcs = [w[1] for w in g]
        pcs = [w[1] for w in p]
        gents = list(set(get_entities(gcs)))
        pents = list(set(get_entities(pcs)))
        # for ((gw, gc), (pw, pc)) in zip(gold, pred):
        #     gents = get_entities(gc)
        #     pents = get_entities(pc)
        for e1 in pents:
            flag = False
            c1 = e1[0]
            e1 = e1[1:]
            for e2 in gents:
                c2 = e2[0]
                e2 = e2[1:]

                if match(e1, e2):
                    pseq.append(c1)
                    gseq.append(c2)
                    if c1 == c2:
                        tp += 1
                        lst.append("tp")
                    else:
                        fp += 1
                        lst.append("fp")
                    flag = True
                    break
            if not flag:
                pseq.append(c1)
                gseq.append("None")
                fp += 1
                lst.append("fp")
        for e1 in gents:
            flag = False
            c1 = e1[0]
            e1 = e1[1:]
            for e2 in pents:
                c2 = e2[0]
                e2 = e2[1:]
                if match(e1, e2):
                    # if c1 != c2:
                    #     gseq.append(c1)
                    #     pseq.append(c2)
                    #     fn += 1
                    #     lst.append('fn')
                    flag = True
                    break
            if not flag:
                gseq.append(c1)
                pseq.append("None")
                fn += 1
                lst.append("fn")
    labels = sorted(set([g for g in gseq if g != "None"]))
    labels.insert(0, "None")

    conf = confusion_matrix(gseq, pseq, labels=labels)
    print(tp, fp, fn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = 2 * p * r / (p + r)
    print(p, r, f)

    def f(a, b):
        if a == b:
            return "tp"
        else:
            if b != "None":
                return "fp"
            else:
                return "fn"

    l = [f(g, p) for g, p in zip(gseq, pseq)]
    for jdg, cjdg, g, p in zip(lst, l, gseq, pseq):
        if jdg != cjdg:
            print(jdg, cjdg, g, p)

    from collections import Counter

    print(Counter(lst))
    print(Counter(l))

    return conf, labels if return_labels else conf


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("gold", type=Path)
    parser.add_argument("pred", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ner", action="store_true")
    args = parser.parse_args()

    if args.ner:
        print("Soft match")
        conf, labels = bio2conf(args.gold, args.pred, strict=False, return_labels=True)
        p, r, f = confusion_matrix2metrics(conf, negative_indices=[labels.index("None")])
        print(confusion_matrix2metrics(conf, negative_indices=[labels.index("None")], return_count=True))
        print("Precision: {}".format(p))
        print("Recall: {}".format(r))
        print("F1: {}".format(f))
        print()
        # draw matrix
        fig = draw_confusion_matrix(conf, labels=labels)
        if args.output:
            fig.savefig("soft.pdf")
        else:
            fig.show()

        print("Strict")
        conf, labels = bio2conf(args.gold, args.pred, strict=True, return_labels=True)
        p, r, f = confusion_matrix2metrics(conf, negative_indices=[labels.index("None")])
        print(confusion_matrix2metrics(conf, negative_indices=[labels.index("None")], return_count=True))
        print("Precision: {}".format(p))
        print("Recall: {}".format(r))
        print("F1: {}".format(f))
        print()
        # draw matrix
        fig = draw_confusion_matrix(conf, labels=labels, figsize=(15, 10), label_fontsize=16, tick_fontsize=13)
        if args.output:
            fig.savefig("exact.pdf")
        else:
            fig.show()

    else:
        dic = {
            "None": 0,
            "Amount_Of": 1,
            "Apparatus_Attr_Of": 2,
            "Apparatus_Of": 3,
            "Atmospheric_Material": 4,
            "Brand_Of": 5,
            "Condition_Of": 6,
            "Coref_Of": 7,
            "Descriptor_Of": 8,
            "Next_Operation": 9,
            "Number_Of": 10,
            "Participant_Material": 11,
            "Property_Of": 12,
            "Recipe_Precursor": 13,
            "Recipe_Target": 14,
            "Solvent_Material": 15,
            "Type_Of": 16,
        }

        gold_data = RelationData(args.gold, pattern="*.ann")
        pred_data = RelationData(args.pred, pattern="*.ann")

        conf = get_confusion_matrix(gold_data, pred_data, labels=list(dic.keys()))
        p, r, f = confusion_matrix2metrics(conf, negative_indices=[dic["None"]])
        print("Precision: {}".format(p))
        print("Recall: {}".format(r))
        print("F1: {}".format(f))

        print(confusion_matrix2metrics(conf, negative_indices=[dic["None"]], return_count=True))

        fig = draw_confusion_matrix(conf, labels=list(dic.keys()))
        if args.output:
            fig.savefig(args.output)
        else:
            fig.show()
