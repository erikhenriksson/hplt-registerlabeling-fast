#!/usr/bin/env python3

import sys
import json

from collections import defaultdict
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

TOP_LEVEL_LABELS = list(LABEL_HIERARCHY.keys())

LABEL_PARENT = {c: p for p, cs in LABEL_HIERARCHY.items() for c in cs}

NO_LABEL = (None, None)  # key for docs with no label

HYBRID_LABEL = ("hybrid", None)

# Represent top-level label p as (p, None) and second-level label c
# with parent p as (p, c) when taking counts.
LABEL_MAP = {t: (t, None) for t in TOP_LEVEL_LABELS}
LABEL_MAP.update({c: (p, c) for c, p in LABEL_PARENT.items()})


def argparser():
    ap = ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("jsonl", nargs="+")
    return ap


def assign_labels(probabilities, threshold):
    labels = set()
    for label, prob in probabilities.items():
        if prob >= threshold:
            labels.add(label)
            if label in LABEL_PARENT:
                # assure that parent also included
                labels.add(LABEL_PARENT[label])
    return labels


def process(fn, stats, args):
    total = 0
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            if args.limit is not None and ln > args.limit:
                break
            d = json.loads(l)
            r = d["register_probabilities"]
            labels = assign_labels(r, args.threshold)
            if not labels:
                stats[NO_LABEL] += 1
            else:
                for label in labels:
                    key = LABEL_MAP[label]
                    stats[key] += 1
            total += 1
    return total


def main(argv):
    args = argparser().parse_args(argv[1:])

    stats, total = defaultdict(int), 0
    for fn in args.jsonl:
        total += process(fn, stats, args)

    # For each top-level label p, fill in implicit "other" 2nd-level
    # entry so that 2nd-level counts add up to top-level count
    for p in TOP_LEVEL_LABELS:
        parent_total = stats.get((p, None), 0)
        if parent_total:  # and LABEL_HIERARCHY[p]:
            child_total = sum(stats.get((p, c), 0) for c in LABEL_HIERARCHY[p])
            stats[(p, "{other}")] = parent_total - child_total

    def label_string(k):
        if k is NO_LABEL:
            return "no register"
        elif k[1] is None:
            return k[0]
        else:
            return f'    {k[0]}, {k[1].strip("{}")}'

    keyf = lambda k: (k[0] or "~", k[1] or "")  # sort order for None
    for k in sorted(stats.keys(), key=keyf):
        print(f"{label_string(k)}: {stats[k]} ({stats[k]/total:.1%})")
    print(f"TOTAL: {total}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
