from pathlib import Path
import spacy
import json


class RelationDatum():
    def __init__(self, path, data_type='auto'):
        self.data_path = Path(path)

        if data_type == 'auto':
            self.data_type = self.data_path.suffix[1:]
        else:
            self.data_type = data_type

        self.data = self.parse()

    def parse(self):
        return self.parse(self.data_path, data_type=self.data_type)

    def parse(self, data_path, data_type='auto'):
        data_path = Path(data_path)
        if data_type == 'auto':
            data_type = data_path.suffix[1:]

        if data_type == 'ann':
            data = self.parse_ann(data_path)
        else:
            raise NotImplementedError

        self.data = data
        return data

    @staticmethod
    def parse_ann(ann_path):
        ann_path = Path(data_path)
        txt_path = data_path.parent / "{}.txt".format(ann_path.stem)

        if not (ann_path.exists() and txt_path.exists()):
            raise FileNotFoundError

        ann_loader = AnnDataLoader()

        data = {}

        text, ents, rels, events = ann_loader.load_ann(ann_path, txt_path)
        data['id'] = ann_path.stem
        data['text'] = text
        data['entity'] = ents
        data['relation'] = rels
        data['event'] = events

        return data

    def export_docred(self, ofile, **kwargs):
        otxt = self.export_docred(**kwargs)
        with open(ofile, 'w') as f:
            f.write(otxt)

        return otxt

    def export_docred(self, *args, **kwargs):
        dic = self.export_docred_as_dict(*args, **kwargs)
        otxt = json.dumps(dic)
        return otxt

    def export_docred_as_dict(self, spacy='en_core_sci_sm'):
        assert self.data
        nlp = spacy.load(spacy)

        dic = {}
        dic['title'] = data['id']
        dic['vertexSet'] = []
        dic['labels'] = []

        doc = nlp(data['text'])

        sents = [list(map(lambda x: x.text, s)) for s in doc.sents]
        dic['sents'] = sents

        # get span of word
        for ent in self.data['entity'].values():
            start = ent['start']
            end = ent['end']
            label = ent['label']
            entity = ent['entity']

            wstart = None
            wend = None
            for sid, sent in enumerate(doc.sents):
                for wid, token in enumerate(sent):
                    if start > token.idx and wstart is None:
                        wstart = wid - 1
                    if end > token.idx and wend is None:
                        assert wstart is not None
                        wend = wid
                        mention = sent[wstart:wend].text
                    if wstart is not None and wend is not None:
                        sent_id = sid
                        break
                if wstart is not None and wend is not None:
                    break

            dic['vertexSet'].append([{
                'name': mention,
                'send_id': sent_id,
                'pos': [wstart, wend],
                'type': label
            }])

        # define relation
        keys = list(data['entity'].keys())

        for rel in data['relation'].values():
            rel_label = rel['label']
            arg1 = rel['arg1']
            arg2 = rel['arg2']

            eid1 = keys.index(arg1)
            eid2 = keys.index(arg2)

            dic['labels'].append({'h': eid1, 't': eid2, 'r': rel})

        return dic


class RelationData():
    def __init__(self,
                 dir_path,
                 pattern='*',
                 data_type='auto',
                 spacy='en_core_sci_sm'):
        self.dir_path = Path(dir_path)
        self.pattern = pattern
        files = self.dir_path.glob(pattern)

        self.data = {}

        for f in files:
            self.data[f] = RelationDatum(f)

    def export_docred(self, spacy='en_core_sci_sm'):
        odir = Path(ofile)
        odata = [
            v.export_docred_as_dict(spacy=spacy) for v in self.data.values()
        ]

        otxt = json.dumps(odata)
        return otxt

    def export_docred(self, ofile, **kwargs):
        ofile = Path(ofile)
        otxt = self.export_docred(**kwargs)
        ofile.write_text(otxt)
        return otxt


# ann data loader
class AnnDataLoader():
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.load_ann(*args, **kwargs)

    def load_ann(self, txt_path, ann_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann = f.read()
        ents, rels, events = self.parse_data(text, ann)
        return text, ents, rels, events

    @staticmethod
    def parse_data(txt_raw, ann_raw):
        rels = OrderedDict()
        ents = OrderedDict()
        events = OrderedDict()
        for line in ann_raw.strip().split('\n'):
            sp = line.split('\t')
            tag = sp[0]
            if 'T' in tag:
                sp_s = sp[1].split(' ')
                label = sp_s[0]
                start = int(sp_s[1])
                end = int(sp_s[2])
                entity = sp[-1]
                ents[tag] = {
                    'label': label,
                    'start': start,
                    'end': end,
                    'entity': entity
                }
                assert txt_raw[start:
                               end] == entity, 'Not matched: span and word'
            elif 'R' in tag:
                # relation
                sp_s = sp[1].split(' ')
                # delete Operatin"-a"
                label = sp_s[0].split('-')[0]
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
                    'label': label,
                    'arg1': arg1,
                    'arg2': arg2,
                }
            elif 'E' in tag:
                if tag not in events.keys():
                    events[tag] = []
                for s in sp[1].split(' '):
                    if 'T' in s:
                        events[tag].append(s.split(':'))
            else:
                pass
        return ents, rels, events