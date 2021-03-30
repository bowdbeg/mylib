from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

def bio2ann(input, output, pattern="*.tsv"):
    input = Path(input)
    output = Path(output)

    if input.is_dir() != output.is_dir():
        raise ValueError

    if input.is_dir():
        inputs = input.glob(pattern)
    else:
        inputs = [input]

    for ifile in tqdm(inputs):
        text = ifile.read_text()
        lines = [line.split("\t") for line in text.split("\n")]

        otxt = ""
        anns = []
        buf = []
        buf_lbl = ""
        pre = "O"
        for line in lines:
            if len(line) < 2:
                otxt += "\n"
                continue
            word = line[0]
            tag = line[1]
            bio = tag[0]
            label = label = tag[2:] if bio in ["B", "I"] else ""

            bufw = " ".join(buf)
            start = len(otxt)
            end = len(otxt) + len(bufw)
            if bio == "B" or (bio == "I" and (pre == "O" or label != buf_lbl)):
                if buf:
                    anns.append((buf_lbl, start, end, bufw))
                    otxt += bufw + " "
                buf = [word]
                buf_lbl = label
            elif bio == "I":
                buf.append(word)
            elif bio == "O":
                if buf:
                    anns.append((buf_lbl, start, end, bufw))
                    otxt += bufw + " "
                otxt += word + " "
                buf = []
                buf_lbl = ""
            else:
                raise NotImplementedError

            pre = bio
        otxt = otxt.strip()
        oann = "\n".join(
            ["T{}\t{} {} {}\t{}".format(i, lbl, st, en, w) for i, (lbl, st, en, w) in enumerate(anns, start=1)]
        )
        (output / (ifile.stem + ".ann")).write_text(oann)
        (output / (ifile.stem + ".txt")).write_text(otxt)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)

    args = parser.parse_args()
    bio2ann(args.input, args.output)
