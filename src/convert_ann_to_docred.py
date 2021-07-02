from typing import Pattern
from relation_data import RelationData
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('input',type=Path)
parser.add_argument('--output','-o',type=Path)
args= parser.parse_args()

data = RelationData(args.input,pattern='*.ann')
if args.output:
    txt = data.export_docred(args.output)
else:
    print(data.export_docred(args.output))