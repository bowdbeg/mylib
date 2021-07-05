from argparse import ArgumentParser
from pathlib import Path
import re
from collections import defaultdict
from timebank import TimebankDatum
import os

label_dic = {
    "a": "AFTER",
    "b": "BEFORE",
    "i": "INCLUDES",
    "ii": "IS_INCLUDED",
    "s": "SIMULTANEOUS",
    "v": "VAGUE",
}

parser = ArgumentParser()
parser.add_argument("dataset_dir", type=Path, help="Dataset directory")
parser.add_argument("ref_file", type=Path, help="Reference file, tsv format")
parser.add_argument("output_dir", type=Path, help="Output directory")
args = parser.parse_args()

if not args.output_dir.exists():
    os.makedirs(args.output_dir)

files = list(args.dataset_dir.glob("*"))

refdata = defaultdict(list)
for line in args.ref_file.read_text().strip().split("\n"):
    sp = line.split()
    assert len(sp) == 4
    refdata[sp[0]].append(sp[1:])

for f in files:
    if not f.stem in refdata:
        continue
    output_path = args.output_dir / f.name
    refdat = refdata[f.stem]
    data = TimebankDatum(f)
    data.reset_relation()
    for d in refdat:
        data.add_relation(d[0],d[1],label_dic[d[2]])
    data.export(output_path)
