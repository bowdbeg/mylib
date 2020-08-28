from pathlib import Path


class RelationDatum():
    def __init__(self, path, data_type='auto'):
        self.data_path = Path(path)

        if data_type == 'auto':
            self.data_type = self.data_path.suffix[1:]
        else:
            self.data_type = data_type

        self.parse()

    def parse(self):
        return self.parse(self.data_path, data_type=self.data_type)

    def parse(self, data_path, data_type='auto'):
        data_path  = Path(data_path)
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
        txt_path = data_path.parent/ "{}.txt".format(ann_path.stem)

        if not(ann_path.exists() and txt_path.exists()):
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
    
    def export_docred(self):



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