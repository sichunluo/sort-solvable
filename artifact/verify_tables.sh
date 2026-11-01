#!/bin/bash
# Consistency test: regenerate every table and figure of the paper from the released results and fail if any
# generated table differs from the one in paper_icde/tables, or if a table of the paper is neither generated nor
# listed below as hand-written. Run from the repository root.
set -e
HANDWRITTEN="protocols.tex"   # the protocol map (Table "protocols"): text only, no numbers
PROC=${SEQREC_PROCESSED:-data/seqrec/processed}
TMP=$(mktemp -d)
SEQREC_PROCESSED="$PROC" python3 -I code/seqrec/make_tables.py data/seqrec/results "$TMP/tables" > "$TMP/tables.log"
# Table "bert" reads summary_strongest.json, which make_tables.py writes
python3 -I code/seqrec/bert_summary.py data/seqrec/results "$PROC" "$TMP/tables" > "$TMP/bert.log"
python3 -I code/seqrec/make_figs.py data/seqrec/results "$PROC" "$TMP/figs" > "$TMP/figs.log"
status=0
for f in paper_icde/tables/*.tex; do
  b=$(basename "$f")
  g="$TMP/tables/$b"
  if [ ! -f "$g" ]; then
    case " $HANDWRITTEN " in
      *" $b "*) echo "HAND    $b" ;;
      *) echo "MISSING $b"; status=1 ;;
    esac
  elif diff -q "$f" "$g" > /dev/null; then echo "OK      $b"; else echo "DIFFERS $b"; status=1; fi
done
exit $status
