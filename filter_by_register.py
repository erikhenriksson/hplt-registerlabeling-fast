#!/usr/bin/env python3

import sys
import json

import zstandard as zstd

from argparse import ArgumentParser

from label_stats import assign_labels


def argparser():
    ap = ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("textfile")
    ap.add_argument("labelfile")
    ap.add_argument("registers", help="R1[,R2...]")
    return ap


def zopen(fn):
    if fn.endswith(".zst"):
        return zstd.open(fn, "rt")
    else:
        return open(fn)


def process(textd, labeld, args):
    assert textd["id"] == labeld["id"], "id mismatch"
    probabilities = labeld["register_probabilities"]
    labels = assign_labels(probabilities, args.threshold)
    if labels & args.registers:
        assert "register_probabilities" not in textd
        textd["register_probabilities"] = probabilities
        print(json.dumps(textd, ensure_ascii=False))


def main(argv):
    args = argparser().parse_args(argv[1:])
    args.registers = set(args.registers.split(","))

    with zopen(args.textfile) as textf:
        with zopen(args.labelfile) as labelf:
            for textl in textf:
                textd = json.loads(textl)
                for labell in labelf:
                    labeld = json.loads(labell)
                    if textd["id"] == labeld["id"]:
                        process(textd, labeld, args)
                        break


if __name__ == "__main__":
    sys.exit(main(sys.argv))
