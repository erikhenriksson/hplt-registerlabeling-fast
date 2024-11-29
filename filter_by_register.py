#!/usr/bin/env python3

import sys
import json

import zstandard as zstd

from argparse import ArgumentParser

LABEL_HIERARCHY = {
    "MT": [],
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}

LABEL_PARENT = {c: p for p, cs in LABEL_HIERARCHY.items() for c in cs}


def is_hybrid(labels):
    if len(labels) > 2:
        return True
    if len(labels) == 2:
        l1, l2 = labels
        return not (
            l1 in LABEL_PARENT
            and LABEL_PARENT[l1] == l2
            or l2 in LABEL_PARENT
            and LABEL_PARENT[l2] == l1
        )
    return False


def assign_labels(probabilities, threshold):
    labels = set()
    for label, prob in probabilities.items():
        if prob >= threshold:
            labels.add(label)
            if label in LABEL_PARENT:
                # assure that parent also included
                labels.add(LABEL_PARENT[label])
    return labels


def process(textd, labeld, args):
    assert textd["id"] == labeld["id"], "id mismatch"
    probabilities = labeld["register_probabilities"]
    labels = assign_labels(probabilities, args.threshold)

    if args.exclude_hybrids and is_hybrid(labels):
        return

    if labels & args.registers:
        assert "register_probabilities" not in textd
        textd["register_probabilities"] = probabilities
        print(json.dumps(textd, ensure_ascii=False))


def argparser():
    ap = ArgumentParser()
    ap.add_argument("textfile")
    ap.add_argument("labelfile")
    ap.add_argument("registers", help="R1[,R2...]")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--exclude_hybrids", action="store_true")
    return ap


def zopen(fn):
    if fn.endswith(".zst"):
        return zstd.open(fn, "rt")
    else:
        return open(fn)


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
