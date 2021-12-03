from argparse import ArgumentParser
from pathlib import Path
import re
from collections import defaultdict
from timebank import TimebankDatum
import os
from tqdm import tqdm

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
parser.add_argument("--matres", action="store_true")
args = parser.parse_args()

if not args.output_dir.exists():
    os.makedirs(args.output_dir)

files = list(args.dataset_dir.glob("*"))

refdata = defaultdict(list)
for line in args.ref_file.read_text().strip().split("\n"):
    sp = line.split()
    # assert len(sp) == 4
    if args.matres:
        d = ["ei{}".format(sp[3]), "ei{}".format(sp[4]), sp[5]]
    else:
        d = sp[1:]
    refdata[sp[0]].append(d)

for f in tqdm(files):
    if not f.stem in refdata:
        continue
    output_path = args.output_dir / f.name
    refdat = refdata[f.stem]
    data = TimebankDatum(f)
    data.reset_relation()
    # print(data.eiid2eid.keys())
    for d in refdat:
        if args.matres:
            if not(d[0] in data.eiid2eid and d[1] in data.eiid2eid):
                print(d[0],d[1])
            else:
                data.add_relation(data.eiid2eid[d[0]], data.eiid2eid[d[1]], d[2])
        else:
            data.add_relation(d[0], d[1], label_dic[d[2]])
    data.export(output_path)
