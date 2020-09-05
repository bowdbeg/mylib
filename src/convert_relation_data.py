from relation_data import RelationData
from pathlib import Path
from argparse import ArgumentParser

TYPES = ["ann", "docred"]
EXTS = {"ann": "ann", "docred": "json"}


def infer_type(path):
    path = Path(path)

    if path.is_dir():
        files = path.glob("*.{}".format(ext))
        out = None
        for tp in TYPES:
            ext = EXTS[tp]
            if any([ext in f.suffix for f in files]):
                assert out is None
                out = tp
    else:
        out = None
        for tp in TYPES:
            ext = EXTS[tp]
            if ext in path.suffix:
                assert out is None
                out = tp

    return out


def convert(args):
    input = args.input
    output = args.output
    output_type = args.output_type
    input_type = args.input_type
    spacy_model = args.spacy

    if input_type == "auto":
        input_type = infer_type(input)
        if input_type is None:
            raise ValueError("Cannot infer type from your input.")

    if input_type == "ann":
        pattern = "*.ann"
    else:
        raise NotImplementedError

    if output_type == "docred":
        data = RelationData(
            input, pattern=pattern, data_type=input_type, spacy_model=spacy_model
        )
        data.export_docred(output, spacy_model=spacy_model)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--output_type", type=str, choices=TYPES, required=True)
    parser.add_argument(
        "--input_type", type=str, choices=["auto", *TYPES], default="auto"
    )
    parser.add_argument("--spacy", type=str, default="en_core_sci_sm")

    args = parser.parse_args()

    convert(args)