#!/usr/bin/env python3

import sys
import re
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import matplotlib.patheffects as path_effects

from argparse import ArgumentParser
from collections import OrderedDict

# regexes for stats lines
TOP_LEVEL_RE = re.compile(r"^([A-Z]{2}): ([0-9]+)")
SECOND_LEVEL_RE = re.compile(r"^ +([A-Z]{2}, [a-z{}]+): ([0-9]+)")
NO_REGISTER_RE = re.compile(r".*?no register.*: ([0-9]+)")
TOTAL_RE = re.compile(r"^TOTAL: ([0-9]+)")

# Define the Seaborn colorblind palette with named colors
colorblind_palette = sns.color_palette("colorblind", 10)
color_names = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "grey",
    "yellow",
    "cyan",
]
color_dict = dict(zip(color_names, colorblind_palette))

# color by label
COLOR_MAP = {
    "no label": "#B8BAC0",  # lighter gray
    "MT": "#6BBC5F",  # brightened green
    "IP": "#F76B6D",  # brightened red
    "IN": "#5E89B7",  # brightened blue
    "LY": "#FFB1BB",  # softened pink
    "SP": "#FFE266",  # brightened yellow
    "ID": "#83C5C0",  # brightened teal
    "NA": "#FF9E45",  # brightened orange
    "HI": "#B38971",  # lightened brown
    "OP": "#C48FB6",  # lightened purple
}


def argparser():
    ap = ArgumentParser()
    ap.add_argument("stats")
    return ap


def parse_stats(fn):
    inner, outer, no_label = OrderedDict(), OrderedDict(), None
    with open(fn) as f:
        for l in f:
            m = TOP_LEVEL_RE.search(l)
            if m:
                label, count = m.groups()
                assert label not in inner
                inner[label] = int(count)
                continue
            m = SECOND_LEVEL_RE.search(l)
            if m:
                label, count = m.groups()
                assert label not in outer
                outer[label] = int(count)
                continue
            m = NO_REGISTER_RE.search(l)
            if m:
                assert no_label is None
                no_label = int(m.group(1))
                continue
            m = TOTAL_RE.search(l)
            if m:
                # print('total', m.groups())
                continue
            raise ValueError(f"failed to parse line: {l}")
    return inner, outer, no_label


def add_labels(ax, wedges, labels, distance):
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        x = distance * wedge.r * np.cos(np.radians(angle))
        y = distance * wedge.r * np.sin(np.radians(angle))
        text = ax.text(
            x, y, labels[i], ha="center", va="center", fontsize=10, color="black"
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=2, foreground="white"),
                path_effects.Normal(),
            ]
        )


def main(argv):
    args = argparser().parse_args(argv[1:])

    inner, outer, no_label = parse_stats(args.stats)

    inner["no label"] = no_label
    outer["no label"] = no_label

    assert sum(inner.values()) == sum(outer.values()), "totals mismatch"

    fig, ax = plt.subplots(figsize=(8, 8))

    inner_labels = list(inner.keys())
    # inner_colors = sns.color_palette('Set2')
    inner_colors = [COLOR_MAP[i] for i in inner_labels]
    outer_colors = [
        inner_colors[inner_labels.index(k.split(",")[0])] for k in outer.keys()
    ]

    inner_wedges, _ = ax.pie(
        inner.values(),
        colors=inner_colors,
        radius=0.6,
    )
    outer_wedges, _ = ax.pie(
        outer.values(),
        labels=outer.keys(),
        colors=outer_colors,
        wedgeprops=dict(width=0.4, edgecolor="white"),
    )

    add_labels(ax, inner_wedges, inner_labels, 0.5)

    plt.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
