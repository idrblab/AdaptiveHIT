#!/usr/bin/bash
set -euo pipefail

# Downloads the large third-party assets listed in README.md's
# "Manual Resource Download" table and puts them where the pipeline expects.
# Idempotent: an asset whose target already exists is skipped, so this is safe
# to re-run after an interrupted download.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="http://47.88.56.212/iTarget"

# marker|url_basename|destination dir (relative to repo root)|archive type
ASSETS=(
    # python.tar.gz (X-Mol's bundled Python 2.7 runtime) is deliberately not
    # fetched: run.sh's `xmol` conda env replaces it.
    "_ForFeatures/xmol/FT_to_embedding/data/model/step_400000/encoder_layer_0_ffn_fc_0.b_0|step_400000_20200326221400.tar|_ForFeatures/xmol/FT_to_embedding/data/model/step_400000|tar"
    "_ForFeatures/esm2/pretrained_esm2_models/esm2_t36_3B_UR50D.pt|esm2_t36_3B_UR50D.pt|_ForFeatures/esm2/pretrained_esm2_models|raw"
    "base_model/ConPLex_dev/models/huggingface|huggingface.tar|base_model/ConPLex_dev/models|tar"
    "base_model/ConPLex_dev/dataset/DUDe/full.tsv|full.tsv|base_model/ConPLex_dev/dataset/DUDe|raw"
    "dataset/Multisimi|dataset.tar|.|tar"
)

for entry in "${ASSETS[@]}"; do
    IFS='|' read -r marker name dest kind <<< "$entry"

    if [ -e "$ROOT/$marker" ]; then
        echo "[skip] $name -- $marker already present"
        continue
    fi

    mkdir -p "$ROOT/$dest"
    echo "[get ] $name -> $dest"
    # Download beside the destination, then extract. The host tends to close
    # long transfers early (curl exit 18), so retry: -C - resumes the partial
    # file, and each outer attempt picks up where the last one stopped.
    tmp="$ROOT/$dest/.$name.part"
    ok=0
    for attempt in $(seq 1 "${PULL_MAX_ATTEMPTS:-60}"); do
        if curl -fsSL -C - --retry 5 --retry-delay 3 --retry-all-errors \
                -o "$tmp" "$BASE_URL/$name"; then
            ok=1; break
        fi
        echo "       attempt $attempt stopped at $(stat -c%s "$tmp" 2>/dev/null || echo 0) bytes; resuming"
        sleep 3
    done
    if [ "$ok" -ne 1 ]; then
        echo "[fail] $name did not finish downloading; re-run to resume." >&2
        exit 1
    fi

    case "$kind" in
        tar)    tar xf "$tmp" -C "$ROOT/$dest" && rm -f "$tmp" ;;
        tar.gz) tar xzf "$tmp" -C "$ROOT/$dest" && rm -f "$tmp" ;;
        raw)    mv "$tmp" "$ROOT/$dest/$name" ;;
    esac
    echo "[ok  ] $name"
done

echo "All external assets are in place."
