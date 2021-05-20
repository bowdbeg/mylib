from relation_data import RelationData
from argparse import ArgumentParser
from pathlib import Path
from collections import Counter, OrderedDict
from statistics import mean

parser = ArgumentParser()
parser.add_argument("gold", type=Path)
parser.add_argument("pred", type=Path)
parser.add_argument("--tex", action="store_true")
args = parser.parse_args()

gold = RelationData(args.gold, pattern="*.ann", fast=True)
pred = RelationData(args.pred, pattern="*.ann", fast=True)


def get_relation_list(data):
    out = OrderedDict()
    for k, v in data.items():
        out[k] = []
        for rel in v["relation"].values():
            arg1 = v["entity"][rel["arg1"]]
            arg2 = v["entity"][rel["arg2"]]
            d = (
                rel["label"],
                arg1["entity"].replace(" ", ""),
                arg1["label"],
                arg2["entity"].replace(" ", ""),
                arg2["label"],
            )
            # d = (rel["label"], arg1["start"], arg1["end"], arg1["label"], arg2["start"], arg2["end"], arg2["label"])
            if not d in out[k]:
                out[k].append(d)
    return out


def get_entity_list(data):
    out = OrderedDict()
    for k, v in data.items():
        out[k] = []
        for ent in v["entity"].values():
            d = (ent['label'], ent["start"], ent["end"])
            # d = (rel["label"], arg1["start"], arg1["end"], arg1["label"], arg2["start"], arg2["end"], arg2["label"])
            if not d in out[k]:
                out[k].append(d)
    return out


def all_match(a, b):
    return all([x == y for x, y in zip(a, b)])


def get_tf(l1, l2, class_index=None):
    tr = 0
    fl = 0
    each = {}
    for d1, d2 in zip(l1, l2):
        for x in d1:
            jdg = any([all_match(x, y) for y in d2])
            tr += jdg
            fl += not jdg
            if not class_index is None:
                if x[class_index] in each:
                    each[x[class_index]]["true"] += jdg
                    each[x[class_index]]["false"] += not jdg
                else:
                    each[x[class_index]] = Counter()
    if not class_index is None:
        return tr, fl, each
    else:
        return tr, fl


gdic = get_entity_list(gold)
pdic = get_entity_list(pred)
plist = [pdic[k] if k in pdic else p for k in gdic.keys()]
glist = list(gdic.values())

tp, fn, cls_fn = get_tf(glist, plist, class_index=0)
tp1, fp, cls_fp = get_tf(plist, glist, class_index=0)
assert tp == tp1

each = {}
for c in cls_fn:
    each[c] = Counter()
    each[c]["tp"] = cls_fn[c]["true"]
    each[c]["fn"] = cls_fn[c]["false"]
for c in cls_fp:
    if not c in each:
        each[c] = Counter()
    # each[c]['tp'] = cls_fp[c]['true']
    each[c]["fp"] = cls_fp[c]["false"]


def harmonic(a, b):
    a = float(a)
    b = float(b)
    return (2 * a * b) / (a + b + 1e-20)


def get_score(tp, fp, fn):
    precision = tp / (fp + tp + 1e-20)
    recall = tp / (fn + tp + 1e-20)
    f1 = harmonic(recall, precision)
    return precision, recall, f1


fs = []
ps = []
rs = []
if args.tex:
    sep = " & "
    end = " \\\\\n"
else:
    sep = "\t"
    end = "\n"
for c in each:
    p, r, f = get_score(each[c]["tp"], each[c]["fp"], each[c]["fn"])
    print(c, "{:.03f}".format(p), "{:.03f}".format(r), "{:.03f}".format(f), sep=sep, end=end)
    fs.append(f)
    rs.append(r)
    ps.append(p)
p, r, f = get_score(tp, fp, fn)
print("micro", "{:.03f}".format(p), "{:.03f}".format(r), "{:.03f}".format(f), sep=sep, end=end)
print("macro", "{:.03f}".format(mean(rs)), "{:.03f}".format(mean(ps)), "{:.03f}".format(mean(fs)), sep=sep, end=end)
