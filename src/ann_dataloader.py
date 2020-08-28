import os
import sys
from collections import OrderedDict
import torch


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
        ents, rels, confs = self.parse_data(text, ann)
        return text, ents, rels, confs

    @staticmethod
    def parse_data(txt_raw, ann_raw):
        rels = OrderedDict()
        ents = OrderedDict()
        corefs = OrderedDict()
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
                if 'Coreference' in label:
                    corefs[tag] = {
                        'label': label,
                        'arg1': arg1,
                        'arg2': arg2,
                    }
                else:
                    rels[tag] = {
                        'label': label,
                        'arg1': arg1,
                        'arg2': arg2,
                    }
            else:
                pass
        return ents, rels, corefs