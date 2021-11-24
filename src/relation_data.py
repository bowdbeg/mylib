from pathlib import Path
from numpy.lib.function_base import iterable
import spacy
import json
from collections import OrderedDict
from tqdm import tqdm
from argparse import ArgumentParser
import re
from difflib import Differ
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Union, Any, Iterator, Dict

DATA_TYPES = {"ann","timebank"}

try:
    nlp_def = spacy.load("en_core_sci_sm")
except:
    nlp_def = spacy.load("en_core_web_sm")


class Instance:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)

    def __getitem__(self, key: Any) -> Any:
        return self.__dict__[key]

    def __setitem__(self, key: Any, val: Any) -> None:
        self.__dict__[key] = val


class RelationDatum:
    def __init__(self, path: Union[str, Path] = None, data_type: str = "auto", fast: bool = False) -> None:
        self.data = OrderedDict()
        self.fast = fast

        if path:
            self.path = Path(path)
            if not self.path.exists():
                raise FileNotFoundError("{} cannot find.".format(path))
            self.load(self.path, data_type)

    def load(self, path: Union[Path, str], data_type: str = "auto"):
        self.path = Path(path)

        if data_type == "auto":
            data_type = self.path.suffix[1:]
        if not data_type in DATA_TYPES:
            raise NotImplementedError("Data loader for {} is not implemented.")
        self.data_type = data_type

        self.data = self._parse(self.path, data_type=self.data_type, nlp=nlp_def)

    @classmethod
    def from_dict(cls, dic: Dict):
        o = cls()
        o.data = dic
        return o

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __setitem__(self, key: Any, val: Any) -> None:
        self.data[key] = val

    def __len__(self) -> int:
        return len(self.data)

    def keys(self) -> Iterator:
        return self.data.keys()

    def items(self) -> Iterator:
        return self.data.items()

    def values(self) -> Iterator:
        return self.data.values()

    def _parse(
        self, data_path: Union[str, Path], data_type: str = "auto", nlp: spacy.lang.en.English = nlp_def
    ) -> Dict:
        if data_type == "ann":
            data = self.parse_ann(data_path, nlp=nlp)
        else:
            raise NotImplementedError

        self.data.update(data)
        if not self.fast:
            self._spacy_process(nlp=nlp)
        return data

    def _spacy_process(self, nlp: spacy.lang.en.English = nlp_def):
        self.doc = nlp(self.data["text"])
        self.set_sentnum()

    def set_sentnum(self):
        ents = self.data["entity"]
        doc = self.data["doc"]
        sent_starts = [s.start_char for s in doc.sents]
        sent_ends = [s.end_char for s in doc.sents]
        for key, ent in ents.items():
            start = ent["start"]
            for i, (st, en) in enumerate(zip(sent_starts, sent_ends)):
                if st <= start < en:
                    snum = i
                    break
            self.data["entity"][key]["sent"] = snum

    def parse_ann(self, ann_path, nlp=nlp_def):
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
        if not self.fast:
            data["doc"] = nlp(text)

        return data

    def export_docred(self, ofile=None, **kwargs):
        dic = self.export_docred_as_dict(**kwargs)
        otxt = json.dumps(dic)

        if ofile:
            Path(ofile).write_text(otxt)

        return otxt

    def export_docred_as_dict(self, nlp=nlp_def):
        assert self.data
        data = self.data

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
            line = "{}\t{} Arg1:{} Arg2:{}\t".format(tag, label, arg1, arg2)
            lines.append(line)
        ann_txt = "\n".join(lines)

        if ofile:
            ann_path = Path(str(ofile) + ".ann")
            txt_path = Path(str(ofile) + ".txt")
            ann_path.write_text(ann_txt)
            txt_path.write_text(text)
        else:
            print("txt:", text, sep="\n")
            print("ann:", ann_txt, sep="\n")

        return ann_txt, text

    def sentencize(self, nlp=spacy.load("en_core_sci_sm")):
        doc = nlp(self.data["text"])
        ret = RelationData()
        for snum, sent in enumerate(doc.sents):
            start_sent = sent.start_char
            end_sent = sent.end_char
            entities = {}

            for tag, ent in self.data["entity"].items():
                start = ent["start"]
                end = ent["end"]
                if start_sent <= start < end_sent:
                    ent["start"] = start - start_sent
                    ent["end"] = end - start_sent if end - start_sent < end_sent else end_sent
                    ent["entity"] = sent.text[ent["start"] : ent["end"]]
                    assert sent.text[ent["start"] : ent["end"]] in ent["entity"]
                    entities[tag] = ent

            relations = {}
            for tag, rel in self.data["relation"].items():
                if rel["arg1"] in entities.keys() and rel["arg2"] in entities.keys():
                    relations[tag] = rel
            name = "{}:{}".format(self.data["id"], snum)
            ret[name] = {"id": name, "text": sent.text, "entity": entities, "relation": relations}
            # TODO event is not implemented
        return ret

    def vis_graph(self, out=None, conf=None):
        # if conf:
        #     config = Path(conf).read_text()
        #     for config.
        #     colors = dict([(lbl, d["color"]) for lbl, d in config["drawing"].items() if "color" in d])
        # else:
        #     colors = {}
        colors = {
            "Amount-Misc": "#65a300",
            "Amount-Unit": "lightblue",
            "Apparatus-Descriptor": "#FE8536",
            "Apparatus-Property-Type": "#01D8C4",
            "Apparatus-Unit": "lightblue",
            "Brand": "#f99b43",
            "Characterization-Apparatus": "#a743f9",
            "Condition-Misc": "#65a300",
            "Condition-Type": "#01D8C4",
            "Condition-Unit": "lightblue",
            "Material": "red",
            "Material-Descriptor": "#FE8536",
            "Meta": "#abaf22",
            "Nonrecipe-Material": "red",
            "Number": "#4795FF",
            "Operation": "lightgreen",
            "Property-Misc": "#65a300",
            "Property-Type": "#01D8C4",
            "Property-Unit": "lightblue",
            "Reference": "grey",
            "Synthesis-Apparatus": "#a743f9",
        }
        ecolors = {
            "Amount_Of": 0,
            "Apparatus_Attr_Of": 0,
            "Apparatus_Of": 0,
            "Atmospheric_Material": 0,
            "Brand_Of": 0,
            "Condition_Of": 0,
            "Coref_Of": 0,
            "Descriptor_Of": 0,
            "Next_Operation": 0,
            "Number_Of": 0,
            "Participant_Material": 0,
            "Property_Of": 0,
            "Recipe_Precursor": 0,
            "Recipe_Target": 0,
            "Solvent_Material": 0,
            "Type_Of": 0,
        }
        cm = plt.get_cmap("jet", len(ecolors))
        for i, k in enumerate(ecolors):
            ecolors[k] = cm(i)

        data = self.data
        ents = OrderedDict([(t, e["entity"]) for t, e in self.data["entity"].items()])
        while True:
            flag = False
            for e in ents:
                if list(ents.values()).count(ents[e]) > 1:
                    ents[e] = ents[e] + "_"
                    flag = True
            if not flag:
                break
        elabels = OrderedDict([(t, e["label"]) for t, e in self.data["entity"].items()])
        nodes = []
        col = []
        for key in ents.keys():
            e = ents[key]
            lbl = elabels[key]
            if lbl in colors:
                nodes.append((e, {"color": colors[lbl]}))
                col.append(colors[lbl])
            else:
                nodes.append((e, {"color": "green"}))
                col.append("green")
        edges = OrderedDict([((ents[r["arg1"]], ents[r["arg2"]]), r["label"]) for r in self.data["relation"].values()])

        num_sent = len(list(self.data["doc"].sents))
        order = [
            list(
                map(
                    lambda x: x[0],
                    sorted(
                        [
                            (nodes[j][0], e["start"])
                            for j, e in enumerate(self.data["entity"].values())
                            if e["sent"] == i
                        ],
                        key=lambda x: x[1],
                    ),
                )
            )
            for i in range(num_sent)
        ]
        position = []
        for i, ent in enumerate(self.data["entity"].values()):
            s = ent["sent"]
            position.append([order[s].index(nodes[i][0]), s])
            # tmp[s] += 1
        position = np.array(position, dtype=np.float)
        # func = lambda x: (1 - x ** 2) ** 0.5
        # func = lambda x: -(x ** 2)
        # func = lambda x: np.sin(x * np.pi)
        func = lambda x: 0
        # func = lambda x: 0.5 * (x%2)
        for i, p in enumerate(position):
            # position[i, 0] = ((p[0] - ps.min()) / (ps.max() - ps.min()) - 0.5) * 2 if ps.max() - ps.min() != 0 else 0.0
            position[i, 0] = (p[0] / len(order[int(p[1])]) - 0.5) * 2
            position[i, 1] = ((num_sent - p[1] - 1 + func(position[i, 0]) + 1.0 * (int(p[0]) % 2)) / num_sent - 0.5) * 2

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(list(edges.keys()))
        # pos = nx.spring_layout(G, k=0.01)
        # pos = nx.spectral_layout(G)
        pos = nx.kamada_kawai_layout(G)
        # pos = nx.rescale_layout(np.array(position))
        # pos = OrderedDict([(n, p) for (n, _), p in zip(nodes, position)])
        # print(pos)

        margin = 0.03
        for _ in range(20):
            flag = False
            for k1, p1 in pos.items():
                for k2, p2 in pos.items():
                    dist = np.linalg.norm(p1 - p2)
                    # if np.abs(p1[1] - p2[1]) < hmargin:
                    #     sign = np.sign(p1[1] - p2[1])
                    #     sign = 1.0 if sign == 0 else sign
                    #     pos[k1][1] = p1[1] + sign * abs(hmargin * 1.01 - abs(p1[1] - p2[1])) / 2
                    #     pos[k2][1] = p2[1] - sign * abs(hmargin * 1.01 - abs(p1[1] - p2[1])) / 2
                    #     flag = True
                    if dist < margin:
                        if dist != 0:
                            pos[k1] = p1 + ((margin - dist) / 2) * (p1 - p2) / dist
                            pos[k2] = p2 - ((margin - dist) / 2) * (p1 - p2) / dist
                        else:
                            pos[k1] = p1 + np.array([0, 1]) * margin / 2
                            pos[k1] = p2 - np.array([0, 1]) * margin / 2
                        flag = True

            if not flag:
                break

        w = 30
        h = int(30 * np.sqrt(2))
        plt.figure(figsize=(w, h))
        nx.draw(
            G,
            pos,
            node_color=col,
            node_shape="s",
            arrowsize=30,
            # connectionstyle="arc3,rad=0.1",
            # edge_cmap=plt.get_cmap("jet"),
            edge_color=[ecolors[v] for v in edges.values()],
        )
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edges, font_size=18)
        nx.draw_networkx_labels(G, pos, font_size=18)
        if out:
            plt.savefig(out)
        else:
            plt.show()
        plt.clf()

    # def export_bio(self, transformers_tokenizer=None,tokenizer=None,ofile=None):
    #     def get_start(text, text_sp):
    #         starts = [0 for _ in  range(len(text_sp))]
    #         for i, w in enumerate(text_sp):
    #             ln = len(transformers_tokenizer.convert_tokens_to_string(w))

    #     for key,val in self.data.items():
    #         ents = val['entity']
    #         text = val['text']
    #         text_sp = tokenizer(text)

    #         for key,ent in ents.items():


class RelationData:
    def __init__(
        self, dir_path=None, pattern="*", data_type="auto", nlp=nlp_def, verbose=False, fast=False
    ):
        self.data = OrderedDict()
        self.fast = fast
        self.verbose = verbose
        if dir_path:
            self.load(
                dir_path=dir_path,
                pattern=pattern,
                data_type=data_type,
                nlp=nlp,
            )

    def __len__(self):
        return len(self.data)

    def load(self, dir_path=None, pattern="*", data_type="auto", nlp=nlp_def):
        self.dir_path = Path(dir_path)
        self.pattern = pattern
        files = list(self.dir_path.glob(pattern))

        self.data = OrderedDict()

        if self.verbose:
            for f in tqdm(files, leave=False, desc="Load"):
                self.data[f.stem] = RelationDatum(path=f, data_type=data_type, fast=self.fast)
        else:
            for f in files:
                self.data[f.stem] = RelationDatum(path=f, data_type=data_type, fast=self.fast)

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

    def update(self, dat):
        self.data.update(dat)

    def export_docred(self, ofile=None, nlp=nlp_def):
        odata = [v.export_docred_as_dict(nlp=nlp) for v in tqdm(self.data.values())]

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

    def sentencize(self, nlp=nlp_def):
        ret = RelationData()
        if self.verbose:
            for key, dat in tqdm(self.data.items(), desc="Sentencize", leave=False):
                d = dat.sentencize(nlp=nlp)
                ret.update(d)
        else:
            for key, dat in self.data.items():
                d = dat.sentencize(nlp=nlp)
                ret.update(d)
        return ret

    def vis_graph(self, out=None, conf=None):
        for name, dat in self.data.items():
            o = out if out is None else Path(out) / (str(name) + ".pdf")
            dat.vis_graph(out=o, conf=conf)


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
                ents[tag] = Instance(
                    **{
                        "label": label,
                        "start": start,
                        "end": end,
                        "entity": entity,
                    }
                )
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
                rels[tag] = Instance(
                    **{
                        "label": label,
                        "arg1": arg1,
                        "arg2": arg2,
                    }
                )
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
                (e["start"], e["end"]) for e in gold[key]["entity"].values() if re.search(label_pattern, e["label"])
            ]
            pents = [
                (e["start"], e["end"]) for e in pred[key]["entity"].values() if re.search(label_pattern, e["label"])
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
    precision = tp / (fp + tp + 1e-10)
    recall = tp / (fn + tp + 1e-10)
    f_measure = 2 * precision * recall / (precision + recall + 1e-10)
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


# if __name__ == "__main__":
#     parser = ArgumentParser(description="calculate score between 2 dir")

#     parser.add_argument("gold", type=Path)
#     parser.add_argument("pred", type=Path)
#     parser.add_argument("--strict", action="store_true", default=False)
#     parser.add_argument("--label", type=str, default=".*")
#     # parser.add_argument('--data_type', type=str,default='auto')

#     args = parser.parse_args()

#     gold_path = args.gold
#     pred_path = args.pred
#     strict = args.strict
#     label = args.label

#     gold_data = RelationData(gold_path, pattern="*.ann", data_type="ann")
#     pred_data = RelationData(pred_path, pattern="*.ann", data_type="ann")

#     pred_data = convert_to_target(pred_data, gold_data)

#     p, r, f = calc_score(gold_data, pred_data, strict=strict, label_pattern=label)
#     print("P: {:.5}".format(p))
#     print("R: {:.5}".format(r))
#     print("F: {:.5}".format(f))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)

    args = parser.parse_args()

    data = RelationData(args.input, pattern="*.ann")
    data.vis_graph(out=args.output, conf=args.config)
