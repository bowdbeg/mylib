import json
import os
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser(description="Apply TACRED split to the dataset.")
parser.add_argument("input_dir", type=Path, help="Path to the TACRED dataset directory.")
parser.add_argument("output_dir", type=Path, help="Path to the output directory.")
parser.add_argument("train_split", type=Path, help="Path to the train split file.")
parser.add_argument("dev_split", type=Path, help="Path to the dev split file.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
train_split = set(args.train_split.read_text().strip().split("\n"))
dev_split = set(args.dev_split.read_text().strip().split("\n"))

train_file = args.input_dir / "train.json"
dev_file = args.input_dir / "dev.json"

train_data = json.loads(train_file.read_text())
# apply split
train_data = [d for d in train_data if d["id"] in train_split]
# save
(args.output_dir / "train.json").write_text(json.dumps(train_data, ensure_ascii=False))

dev_data = json.loads(dev_file.read_text())
# apply split
dev_data = [d for d in dev_data if d["id"] in dev_split]
# save
(args.output_dir / "dev.json").write_text(json.dumps(dev_data, ensure_ascii=False))

# copy test file
import shutil

test_file = args.input_dir / "test.json"
shutil.copy(test_file, args.output_dir / "test.json")
