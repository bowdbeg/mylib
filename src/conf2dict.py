import json
from configparser import ConfigParser
from argparse import ArgumentParser
from pathlib import Path


def conf2dict(config_file):
    config = ConfigParser()
    config.read(config_file)
    relation_classes = config["relation"]["classes"].replace(" ", "").split(",")
    entity_classes = ["ROOT", *config["entity"]["classes"].replace(" ", "").split(",")]

    ent_dic = {}
    rel_dic = {}

    for i, e in enumerate(entity_classes):
        ent_dic[e] = i

    for i, r in enumerate(relation_classes):
        rel_dic[r] = i

    return ent_dic, rel_dic


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("config_file")
    parser.add_argument("--ent_json", type=Path)
    parser.add_argument("--rel_json", type=Path)

    args = parser.parse_args()

    ent_dic, rel_dic = conf2dict(args.config_file)
    if args.ent_json:
        with open(args.ent_json, "w") as f:
            json.dump(ent_dic, f)
    else:
        print(ent_dic)

    if args.rel_json:
        with open(args.rel_json, "w") as f:
            json.dump(rel_dic, f)
    else:
        print(rel_dic)