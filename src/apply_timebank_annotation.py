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
parser.add_argument('output_dir',type=Path,help='Output directory')
args = parser.parse_args()

if not args.output_dir.exists():
    os.makedirs(args.output_dir)

files = list(args.dataset_dir.glob("*"))

refdata = defaultdict(list)
for line in args.ref_file.read_text().strip().split("\n"):
    sp = line.split()
    assert len(sp) == 3
    refdata[sp[0]].append(sp[1:])

for f in files:
    if not f.stem in refdata:
        continue
    output_path = args.output_dir/f.name
    cnt = 0
    txt = f.read_text()
    txt = re.sub("^<.?LINK.*$", "", txt)
    refdat = refdata[f.stem]
    data = TimebankDatum(f)
    lines = []
    for d in refdat:
        cnt += 1
        hid = data["instance"][d[1]]["eiid"]
        tid = data["instance"][d[2]]["eiid"]
        hnm = "eventInstanceID" if d[1][0] == "e" else "timeID"
        tnm = "relatedToEventInstance" if d[1][0] == "e" else "relatedToTime"

        l = '<TLINK lid="{}" relType="{}" {}="{}" {}="{}" />'.format(
            cnt, label_dic[d[-1]], hnm, hid, tnm, tid
        )
        lines.append(l)
    txt += '\n'.join(lines)
    output_path.write_text(txt)
