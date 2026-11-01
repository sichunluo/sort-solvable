#!/bin/bash
# Tie audit of the benchmark files and their sources (Table "audit"). Downloads public files into <dir> (one folder
# per source, read as data only) and writes one JSON line per source to <out>.
# Usage: run_audit.sh <download dir> <raw dir of this repo> <out.jsonl>
set -e
D=$1; RAW=$2; OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
get() { mkdir -p "$D/$1"; [ -s "$D/$1/$(basename "$2")" ] || curl -sfL -o "$D/$1/$(basename "$2")" "$2"; }
get amazon14 http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty.json.gz
get amazon14 http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz
get sasrec https://raw.githubusercontent.com/kang205/SASRec/master/data/Beauty.txt
for f in Luxury_Beauty_5 Toys_and_Games_5 Video_Games_5; do
  get amazon18 https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/$f.json.gz
done
get amazon23 https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Toys_and_Games.csv.gz
a() { python3 -I "$HERE/audit_ties.py" "$@" | tail -1 >> "$OUT"; }
: > "$OUT"
for c in Beauty Sports_and_Outdoors Toys_and_Games; do a "Amazon 2014 5-core $c" amazon14 "$RAW/amazon5/reviews_${c}_5.json.gz"; done
a "Amazon 2014 5-core Video Games" amazon14 "$D/amazon14/reviews_Video_Games_5.json.gz"
a "Amazon 2014 Beauty via SASRec preprocessing" sasrec "$D/amazon14/reviews_Beauty.json.gz::$D/sasrec/Beauty.txt"
a "Amazon 2018 5-core Luxury Beauty" amazon18 "$D/amazon18/Luxury_Beauty_5.json.gz"
a "Amazon 2018 5-core Toys" amazon18 "$D/amazon18/Toys_and_Games_5.json.gz"
a "Amazon 2018 5-core Video Games" amazon18 "$D/amazon18/Video_Games_5.json.gz"
a "Amazon 2023 5-core Toys" amazon23 "$D/amazon23/Toys_and_Games.csv.gz"
a "Amazon 2023 5-core Toys (by day)" amazon23 "$D/amazon23/Toys_and_Games.csv.gz" 86400
a "MovieLens-1M" movielens "$RAW/ml/ml-1m.zip::ml-1m/ratings.dat"
a "MovieLens-20M" movielens "$RAW/ml/ml-20m.zip::ml-20m/ratings.csv"
# Code bases that ship the S3-Rec files: compare SHA-256 with the S3-Rec release.
for spec in "Woeee/FMLP-Rec master data" "salesforce/ICLRec master data" "YChen1993/CoSeRec main data" \
            "yehjin-shin/BSARec main src/data"; do
  set -- $spec
  for n in Beauty Sports_and_Outdoors Toys_and_Games; do
    [ "$1" = Woeee/FMLP-Rec ] && [ $n != Beauty ] && continue
    get "repos/${1//\//_}" "https://raw.githubusercontent.com/$1/$2/$3/$n.txt"
    same=$([ "$(shasum -a 256 < "$D/repos/${1//\//_}/$n.txt")" = "$(shasum -a 256 < "$RAW/s3rec/$n.txt")" ] && echo true || echo false)
    echo "{\"repo\": \"$1\", \"file\": \"$n.txt\", \"identical_to_s3rec\": $same}" >> "$OUT"
  done
done
