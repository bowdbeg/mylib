from pathlib import Path
import spacy
import json
from collections import OrderedDict
from tqdm import tqdm
from argparse import ArgumentParser
import re
from difflib import Differ


class RelationDatum:
    def __init__(self, path=None, data_type="auto"):
        self.data = OrderedDict()
        if path:
            self.load(path, data_type)

    def load(self, path, data_type="auto"):
        self.data_path = Path(path)

        if data_type == "auto":
            self.data_type = self.data_path.suffix[1:]
        else:
            self.data_type = data_type

        self.data = self.parse(self.data_path, data_type=self.data_type)
        return self

    def from_dict(self, dic):
        self.data = dic
        return self

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = val

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def parse(self, data_path, data_type="auto"):
        data_path = Path(data_path)
        if data_type == "auto":
            data_type = data_path.suffix[1:]

        if data_type == "ann":
            data = self.parse_ann(data_path)
        else:
            raise NotImplementedError

        self.data = data
        return data

    @staticmethod
    def parse_ann(ann_path):
        ann_path = Path(ann_path)
        txt_path = ann_path.parent / "{}.txt".format(ann_path.stem)

        if not (ann_path.exists() and txt_path.exists()):
            raise FileNotFoundError

        ann_loader = AnnDataLoader()

        data = OrderedDict()

        text, ents, rels, events = ann_loader.load_ann(ann_path, txt_path)
        data["id"] = ann_path.stem
        data["text"] = text
        data["entity"] = ents
        data["relation"] = rels
        data["event"] = events

        return data

    def export_docred(self, ofile=None, **kwargs):
        dic = self.export_docred_as_dict(**kwargs)
        otxt = json.dumps(dic)

        if ofile:
            Path(ofile).write_text(otxt)

        return otxt

    def export_docred_as_dict(self, spacy_model="en_core_sci_sm"):
        assert self.data
        data = self.data
        nlp = spacy.load(spacy_model)

        dic = OrderedDict()
        dic["title"] = data["id"]
        dic["vertexSet"] = []
        dic["labels"] = []

        doc = nlp(data["text"])

        sent_starts = [s[0].idx for s in doc.sents]
        doc_sents = list(doc.sents)
        to_join = []
        for ent in data["entity"].values():
            for idx, start in enumerate(sent_starts):
                if ent["start"] < start <= ent["end"]:
                    flag = False
                    for i in range(len(to_join)):
                        if idx in to_join[i] or idx - 1 in to_join[i]:
                            to_join[i].append(idx)
                            flag = True
                    if not flag:
                        to_join.append([idx, idx - 1])

        joined_sents = []
        for i, s in enumerate(doc_sents):
            start = s.start
            end = s.end
            flag = False
            for idxs in to_join:
                if i in idxs:
                    start = min([doc_sents[idx].start for idx in idxs])
                    end = max([doc_sents[idx].end for idx in idxs])
                    flag = s.start == start
                    break
            if not flag:
                joined_sents.append(doc[start:end])

        sents = [list(map(lambda x: x.text, s)) for s in joined_sents]
        dic["sents"] = sents

        # get span of word
        for ent in data["entity"].values():
            start = ent["start"]
            end = ent["end"]
            label = ent["label"]
            entity = ent["entity"]

            wstart = None
            wend = None
            last_wid = -1
            # TODO if sentencized between entity, join sentence
            for sid, sent in enumerate(joined_sents):
                for wid, token in enumerate(sent):
                    if start < token.idx and wstart is None:
                        if wid != 0:
                            wstart = wid - 1
                            sent_id = sid
                        else:
                            wstart = last_wid
                            sent_id = sid - 1
                    if end <= token.idx and wend is None:
                        assert wstart is not None
                        if wid != 0:
                            wend = wid
                        else:
                            wend = last_wid + 1
                        break
                    last_wid = wid
                if wstart and wend:
                    break

            if wstart is None:
                wstart = last_wid
                wend = last_wid + 1
                sent_id = sid
            elif wend is None:
                wend = last_wid + 1

            assert wstart is not None and wend is not None

            phrase = list(joined_sents)[sent_id][wstart:wend].text
            assert entity in phrase
            dic["vertexSet"].append(
                [
                    {
                        "name": phrase,
                        "sent_id": sent_id,
                        "pos": [wstart, wend],
                        "type": label,
                    }
                ]
            )

        # define relation
        keys = list(data["entity"].keys())

        for rel in data["relation"].values():
            rel_label = rel["label"]
            arg1 = rel["arg1"]
            arg2 = rel["arg2"]

            eid1 = keys.index(arg1)
            eid2 = keys.index(arg2)

            dic["labels"].append({"h": eid1, "t": eid2, "r": rel_label, "evidence": []})

        return dic

    def export_ann(self, ofile=None):
        text = self.data["text"]
        ofile = Path(ofile)
        lines = []
        for tag, val in self.data["entity"].items():
            start = val["start"]
            end = val["end"]
            entity = val["entity"]
            label = val["label"]
            line = "{}\t{} {} {}\t{}".format(tag, label, start, end, entity)
            lines.append(line)
        for tag, val in self.data["relation"].items():
            label = val["label"]
            arg1 = val["arg1"]
            arg2 = val["arg2"]
            line = "{}\t{} Arg1:{} Arg2:{}"
            lines.append(line)
        ann_txt = "\n".join(lines)

        if ofile:
            ann_path = ofile.parent / (ofile.stem + ".ann")
            txt_path = ofile.parent / (ofile.stem + ".txt")
            ann_path.write_text(ann_txt)
            txt_path.write_text(text)
        else:
            print("txt:", text, sep="\n")
            print("ann:", ann_txt, sep="\n")

        return ann_txt, text


class RelationData:
    def __init__(
        self, dir_path=None, pattern="*", data_type="auto", spacy_model="en_core_sci_sm"
    ):
        self.data = OrderedDict()
        if dir_path:
            self.load(
                dir_path=dir_path,
                pattern=pattern,
                data_type=data_type,
                spacy_model=spacy_model,
            )

    def load(
        self, dir_path=None, pattern="*", data_type="auto", spacy_model="en_core_sci_sm"
    ):
        self.dir_path = Path(dir_path)
        self.pattern = pattern
        files = self.dir_path.glob(pattern)

        self.data = OrderedDict()

        for f in files:
            self.data[f.stem] = RelationDatum(path=f, data_type=data_type)

    def from_dict(dic):
        self.data = dic

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = val

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def export_docred(self, ofile=None, spacy_model="en_core_sci_sm"):
        odata = [
            v.export_docred_as_dict(spacy_model=spacy_model)
            for v in tqdm(self.data.values())
        ]

        otxt = json.dumps(odata)
        if ofile:
            ofile = Path(ofile)
            ofile.write_text(otxt)
        return otxt

    def export_ann(self, ofile=None):
        ofile = Path(ofile)
        ann_txts = []
        txt = []
        for k, v in self.data.items():
            a, t = v.export_ann(ofile / k)
            ann_txts.append(a)
            txt.append(t)
        return ann_txts, txt


# ann data loader
class AnnDataLoader:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.load_ann(*args, **kwargs)

    def load_ann(self, ann_path, txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = f.read()
        ents, rels, events = self.parse_data(ann, text)
        return text, ents, rels, events

    @staticmethod
    def parse_data(ann_raw, txt_raw):
        rels = OrderedDict()
        ents = OrderedDict()
        events = OrderedDict()
        for line in ann_raw.strip().split("\n"):
            sp = line.split("\t")
            tag = sp[0]
            if "T" in tag:
                sp_s = sp[1].split(" ")
                label = sp_s[0]
                start = int(sp_s[1])
                end = int(sp_s[2])
                entity = sp[-1]
                ents[tag] = {
                    "label": label,
                    "start": start,
                    "end": end,
                    "entity": entity,
                }
                # assert txt_raw[start:end] == entity, "Not matched: span and word"
            elif "R" in tag:
                # relation
                sp_s = sp[1].split(" ")
                # delete Operatin"-a"
                label = sp_s[0].split("-")[0]
                arg1 = sp_s[1][5:]
                arg2 = sp_s[2][5:]
                # if 'Coref_Of' in label:
                #     corefs[tag] = {
                #         'label': label,
                #         'arg1': arg1,
                #         'arg2': arg2,
                #     }
                # else:
                rels[tag] = {
                    "label": label,
                    "arg1": arg1,
                    "arg2": arg2,
                }
            elif "E" in tag:
                if tag not in events.keys():
                    events[tag] = []
                for s in sp[1].split(" "):
                    if "T" in s:
                        events[tag].append(s.split(":"))
            else:
                pass
        return ents, rels, events


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


def calc_score(gold, pred, mode="entity", strict=True, label_pattern="*"):
    and_keys = [g for g in gold.keys() if g in pred.keys()]
    tp = 0
    fp = 0
    fn = 0
    for key in and_keys:
        if "entity":
            gents = [
                (e["start"], e["end"])
                for e in gold[key]["entity"].values()
                if re.search(label_pattern, e["label"])
            ]
            pents = [
                (e["start"], e["end"])
                for e in pred[key]["entity"].values()
                if re.search(label_pattern, e["label"])
            ]
            for p in pents:
                judges = (
                    [p[0] == g[0] and p[1] == g[1] for g in gents]
                    if strict
                    else [g[0] <= p[0] <= g[1] or g[0] <= p[1] <= g[1] for g in gents]
                )
                if any(judges):
                    tp += 1
                else:
                    fp += 1
            for g in gents:
                judges = (
                    [p[0] == g[0] and p[1] == g[1] for p in pents]
                    if strict
                    else [p[0] <= g[0] <= p[1] or p[0] <= g[1] <= p[1] for p in pents]
                )
                if not any(judges):
                    fn += 1
        elif "relation":
            raise NotImplementedError
    precision = tp / (fp + tp)
    recall = tp / (fn + tp)
    f_measure = 2 * precision * recall / (precision + recall)
    return precision, recall, f_measure


def convert_to_target_datum(datum, target):
    new_data = {}
    new_data["text"] = target["text"]
    new_data["relation"] = datum["relation"]
    new_data["id"] = datum["id"]
    new_data["entity"] = {}
    new_data["event"] = datum["event"]
    new_data["relation"] = datum["relation"]

    diff_indices = diff_string(datum["text"], target["text"])
    for tag, ent in datum["entity"].items():
        start = diff_indices[ent["start"]]
        try:
            end = diff_indices[ent["end"]]
        except IndexError:
            end = start + ent["end"] - ent["start"]
            assert datum["text"][start:end] == target["text"][ent["start"] : ent["end"]]
        t = datum["text"][start:end]
        e = {"start": start, "end": end, "label": ent["label"], "entity": t}
        new_data["entity"][tag] = e
    dat = RelationDatum()
    dat = dat.from_dict(new_data)

    return dat


def convert_to_target(data, target):
    and_keys = [g for g in data.keys() if g in target.keys()]
    for key in and_keys:
        if data[key]["text"] != target[key]["text"]:
            data[key] = convert_to_target_datum(data[key], target[key])
    return data


if __name__ == "__main__":
    parser = ArgumentParser(description="calculate score between 2 dir")

    parser.add_argument("gold", type=Path)
    parser.add_argument("pred", type=Path)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--label", type=str, default=".*")
    # parser.add_argument('--data_type', type=str,default='auto')

    args = parser.parse_args()

    gold_path = args.gold
    pred_path = args.pred
    strict = args.strict
    label = args.label

    gold_data = RelationData(gold_path, pattern="*.ann", data_type="ann")
    pred_data = RelationData(pred_path, pattern="*.ann", data_type="ann")

    pred_data = convert_to_target(pred_data, gold_data)

    p, r, f = calc_score(gold_data, pred_data, strict=strict, label_pattern=label)
    print("P: {:.5}".format(p))
    print("R: {:.5}".format(r))
    print("F: {:.5}".format(f))
