#!/bin/bash
# =====================================================================
#  Download the real EndoNeRF 'pulling_soft_tissues' clip into the layout
#  EndoGaussian expects:  data/endonerf/pulling/{images,depth,masks}/*.png
#                         data/endonerf/pulling/poses_bounds.npy
#
#  Why this exists: the official data is only on the authors' Google Drive,
#  and `gdown` (folder or API path) trips Drive's anonymous daily download
#  cap after a few dozen files. The plain `uc?export=download` endpoint via
#  curl is on a less-restricted path and works for these small (~20 KB) files.
#  File IDs are pinned in tools/endonerf_pulling/fileids.tsv.
#
#  Usage:  bash tools/download_endonerf_pulling.sh
#  Resumable: re-run to fetch only what's missing/corrupt.
# =====================================================================
set -u
cd "$(dirname "$0")/.."                       # repo root
MANIFEST=tools/endonerf_pulling/fileids.tsv
DEST=data/endonerf/pulling

ok=0; miss=0
while IFS=$'\t' read -r sub fn fid; do
    case "$sub" in '#'*|'') continue;; esac    # skip comments/blank
    dir="$DEST/$sub"; mkdir -p "$dir"
    out="$dir/$fn"
    # skip if already a non-trivial file
    if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 100 ]; then ok=$((ok+1)); continue; fi
    for attempt in 1 2 3 4 5; do
        curl -sSL "https://drive.google.com/uc?export=download&id=${fid}" -o "$out"
        if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 100 ]; then break; fi
        sleep $((attempt * 3))
    done
    if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 100 ]; then
        ok=$((ok+1))
    else
        echo "FAILED: $sub/$fn"; miss=$((miss+1))
    fi
    sleep 0.3
done < "$MANIFEST"

echo "done: $ok present, $miss missing"
echo "counts -> images=$(ls $DEST/images/*.png 2>/dev/null|wc -l) depth=$(ls $DEST/depth/*.png 2>/dev/null|wc -l) masks=$(ls $DEST/masks/*.png 2>/dev/null|wc -l) poses=$(ls $DEST/poses_bounds.npy 2>/dev/null|wc -l)"
[ "$miss" -eq 0 ] || exit 1
