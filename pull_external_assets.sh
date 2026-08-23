#!/usr/bin/bash
set -euo pipefail

# Downloads the large third-party assets listed in README.md's
# "Manual Resource Download" table and puts them where the pipeline expects.
# Idempotent: an asset whose target already exists is skipped, so this is safe
# to re-run after an interrupted download.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="http://47.88.56.212/adaptivehit"

# $BASE_URL is a plain static file server -- no CDN, no checksums, no
# versioning -- so an asset can simply stop being served, as three of the five
# below did from the previous /iTarget path. Where the artefact has an
# authoritative upstream, that URL is recorded as a fallback and tried when
# $BASE_URL cannot supply it; README.md lists the upstream first for the same
# reason. Two entries have no fallback URL here: the X-Mol weights (upstream
# is a OneDrive share, see README.md -- not fetchable non-interactively) and
# dataset.tar (benchmark splits released with this manuscript).
#
# marker|url_basename|destination dir (relative to repo root)|archive type|upstream fallback URL
ASSETS=(
    # python.tar.gz (X-Mol's bundled Python 2.7 runtime) is deliberately not
    # fetched: run.sh's `xmol` conda env replaces it.
    "_ForFeatures/xmol/FT_to_embedding/data/model/step_400000/encoder_layer_0_ffn_fc_0.b_0|step_400000_20200326221400.tar|_ForFeatures/xmol/FT_to_embedding/data/model/step_400000|tar|"
    # Meta's own copy: byte-identical to the mirror (both 5678116398 bytes).
    "_ForFeatures/esm2/pretrained_esm2_models/esm2_t36_3B_UR50D.pt|esm2_t36_3B_UR50D.pt|_ForFeatures/esm2/pretrained_esm2_models|raw|https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt"
    # No upstream tarball: ProtBert lives on the HuggingFace Hub as a repo,
    # not an archive. Nothing is lost -- ConPLex_dev/src/featurizers/protein.py
    # fetches Rostlab/prot_bert from the Hub when this directory is absent.
    "base_model/ConPLex_dev/models/huggingface|huggingface.tar|base_model/ConPLex_dev/models|tar|"
    # The URL ConPLex's own dataset/DUDe/download_full_tsv.sh uses.
    "base_model/ConPLex_dev/dataset/DUDe/full.tsv|full.tsv|base_model/ConPLex_dev/dataset/DUDe|raw|https://cb.csail.mit.edu/conplex/data/full.tsv"
    "dataset/Multisimi|dataset.tar|.|tar|"
)

FAILED=()

for entry in "${ASSETS[@]}"; do
    IFS='|' read -r marker name dest kind upstream <<< "$entry"

    if [ -e "$ROOT/$marker" ]; then
        echo "[skip] $name -- $marker already present"
        continue
    fi

    sources=("$BASE_URL/$name")
    [ -n "$upstream" ] && sources+=("$upstream")

    mkdir -p "$ROOT/$dest"
    tmp="$ROOT/$dest/.$name.part"
    ok=0
    for url in "${sources[@]}"; do
        # Probe before downloading. A 404 means that host does not have the
        # file, which no amount of retrying fixes -- without this check the
        # resume loop below burns every attempt re-printing the same error.
        # Anything else (including a server that rejects HEAD) falls through
        # to the loop, which is what handles genuine transient failures.
        code=$(curl -sIL -o /dev/null -w '%{http_code}' --max-time 30 "$url" || echo 000)
        if [ "$code" = "404" ]; then
            echo "[gone] $url -- HTTP 404"
            continue
        fi

        echo "[get ] $name <- $url"
        # Download beside the destination, then extract. The host tends to
        # close long transfers early (curl exit 18), so retry: -C - resumes
        # the partial file, and each outer attempt picks up where the last
        # one stopped.
        for attempt in $(seq 1 "${PULL_MAX_ATTEMPTS:-60}"); do
            if curl -fsSL -C - --retry 5 --retry-delay 3 --retry-all-errors \
                    -o "$tmp" "$url"; then
                ok=1; break
            fi
            echo "       attempt $attempt stopped at $(stat -c%s "$tmp" 2>/dev/null || echo 0) bytes; resuming"
            sleep 3
        done
        [ "$ok" -eq 1 ] && break

        # Falling through to another host: discard the partial file. `-C -`
        # would otherwise resume at that byte offset against a different
        # server, silently splicing two files together.
        echo "       giving up on $url"
        rm -f "$tmp"
    done
    if [ "$ok" -ne 1 ]; then
        # Carry on with the remaining assets rather than aborting: one
        # unreachable file should not stop the others from being fetched.
        echo "[fail] $name could not be fetched from any source." >&2
        FAILED+=("$name (no source served it)")
        continue
    fi

    case "$kind" in
        tar)    tar xf "$tmp" -C "$ROOT/$dest" && rm -f "$tmp" ;;
        tar.gz) tar xzf "$tmp" -C "$ROOT/$dest" && rm -f "$tmp" ;;
        raw)    mv "$tmp" "$ROOT/$dest/$name" ;;
    esac
    echo "[ok  ] $name"
done

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All external assets are in place."
    exit 0
fi

echo
echo "Not fetched:"
printf '  - %s\n' "${FAILED[@]}"
cat <<'EOF'

Every source for these returned 404 or failed, so re-running will not help by
itself. What each one costs, and what you can still do about it:

  huggingface.tar  ProtBert weights for ConPLex. NOT BLOCKING: there is no
                   upstream tarball because ProtBert is a HuggingFace repo,
                   not an archive -- and ConPLex_dev/src/featurizers/protein.py
                   already falls back to fetching Rostlab/prot_bert from the
                   Hub when this directory is absent. Prediction still runs,
                   given access to huggingface.co.

  full.tsv         DUD-E data, only needed to retrain ConPLex from scratch.
                   Upstream is ConPLex's own host and is tried automatically;
                   its own downloader is
                   base_model/ConPLex_dev/dataset/DUDe/download_full_tsv.sh

  dataset.tar      Benchmark CPI splits built for the manuscript. No upstream
                   exists -- they are not a redistribution of a public
                   dataset. Only needed to reproduce the paper's experiments;
                   dataset/toy_dataset/ ships in the repository and is enough
                   for both Quick Start flows in README.md.

  step_400000_*    X-Mol weights. THIS ONE IS BLOCKING -- without it no
                   compound embeddings can be computed. Upstream is the
                   OneDrive share linked from https://github.com/bm2-lab/x-mol
                   ("Pre_trained X-MOL"), which has to be fetched by hand;
                   see the Manual Resource Download table in README.md.
EOF
exit 1
