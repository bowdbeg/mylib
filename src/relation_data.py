from pathlib import Path
import spacy
import json
from collections import OrderedDict
from tqdm import tqdm


class RelationDatum:
    def __init__(self, path=None, data_type="auto"):
        if path:
            self.load(path, data_type)

    def load(self, path, data_type="auto"):
        self.data_path = Path(path)

        if data_type == "auto":
            self.data_type = self.data_path.suffix[1:]
        else:
            self.data_type = data_type

        self.data = self.parse(self.data_path, data_type=self.data_type)

    def from_dict(self, dic):
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
        for tag, val in data["entity"].items():
            start = val["start"]
            end = val["end"]
            entity = val["entity"]
            label = val["label"]
            line = "{}\t{} {} {}\t{}".format(tag, label, start, end, entity)
            lines.append(line)
        for tag, val in dat["relation"].items():
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
        if dir_path:
            self.load(
                dir_path=dir_path,
                pattern=pattern,
                data_type=data_type,
                spacy_model=spacy_model,
            )

    def load(
        dir_path=None, pattern="*", data_type="auto", spacy_model="en_core_sci_sm"
    ):
        self.dir_path = Path(dir_path)
        self.pattern = pattern
        files = self.dir_path.glob(pattern)

        self.data = OrderedDict()

        for f in files:
            self.data[f] = RelationDatum(f)

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
                assert txt_raw[start:end] == entity, "Not matched: span and word"
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