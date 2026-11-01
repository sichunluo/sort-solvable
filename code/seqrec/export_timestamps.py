"""Reconstructed timestamps of the S3-Rec Amazon sequences. Rebuilds the sequences from the 2014 5-core review files
as the S3-Rec preprocessing did (user k = k-th reviewer in file order, items numbered by first appearance, stable
sort on the day-resolution review time), checks that every published sequence is reproduced, and writes one row per
event in the order of <Name>.txt: S3-Rec user id, position in the sequence, S3-Rec item id, ASIN, unixReviewTime.
Same-day events share a timestamp; their order within the day is the ASIN order of the raw file.
Usage: python export_timestamps.py <raw dir> <out dir>   (writes <out dir>/<Name>_timestamps.tsv.gz)
"""
import gzip
import json
import os
import sys

RAW, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
for name in ("Beauty", "Sports_and_Outdoors", "Toys_and_Games"):
    users, order = {}, []
    for line in gzip.open(f"{RAW}/amazon5/reviews_{name}_5.json.gz", "rt"):
        r = json.loads(line)
        if r["reviewerID"] not in users:
            users[r["reviewerID"]] = []
            order.append(r["reviewerID"])
        users[r["reviewerID"]].append((r["asin"], int(r["unixReviewTime"])))
    item2id, rows = {}, []
    for k, u in enumerate(order, 1):
        for pos, (a, t) in enumerate(sorted(users[u], key=lambda x: x[1])):
            item2id.setdefault(a, len(item2id) + 1)
            rows.append((k, pos, item2id[a], a, t))
    seqs = {}
    for k, _, i, _, _ in rows:
        seqs.setdefault(k, []).append(str(i))
    s3 = [l.split() for l in open(f"{RAW}/s3rec/{name}.txt")]
    assert [l[1:] for l in s3] == [seqs[int(l[0])] for l in s3], "S3-Rec sequences not reproduced"
    with gzip.open(f"{OUT}/{name}_timestamps.tsv.gz", "wt") as f:
        f.write("user\tposition\titem\tasin\tunixReviewTime\n")
        f.writelines(f"{k}\t{p}\t{i}\t{a}\t{t}\n" for k, p, i, a, t in rows)
    ties = sum(rows[j][0] == rows[j - 1][0] and rows[j][4] == rows[j - 1][4] for j in range(1, len(rows)))
    print(name, len(s3), "sequences reproduced,", len(rows), "events,", ties, "same-day consecutive pairs", flush=True)
