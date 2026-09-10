#!/bin/bash
# Fetch and unpack the raw trace tarballs into data/<subject>/traces/.
#
#   TRACES_URL=<base url of the release assets> bash tools/fetch_traces.sh [openpilot|transfuser|both]
#
# Expects <base url>/counterfactual-harm-oracle-traces-<subject>.tar as
# produced by tools/pack_traces.sh. Nothing in analysis/ needs the traces;
# they are required only to rebuild data/ from scratch
# (campaign/build_features.py) or to run the CriMe toolbox on the
# reference executions (campaign/run_crime_baselines.py).
set -euo pipefail
WHICH=${1:-both}
HERE=$(cd "$(dirname "$0")/.." && pwd)
: "${TRACES_URL:?set TRACES_URL to the directory holding the tarballs}"
for s in openpilot transfuser; do
  if [ "$WHICH" = "both" ] || [ "$WHICH" = "$s" ]; then
    f="counterfactual-harm-oracle-traces-$s.tar"
    echo "== $s: $TRACES_URL/$f"
    curl -L -o "/tmp/$f" "$TRACES_URL/$f"
    tar -xf "/tmp/$f" -C "$HERE/data"
    rm -f "/tmp/$f"
    echo "   $(find "$HERE/data/$s/traces" -name '*.npz' | wc -l) traces under data/$s/traces"
  fi
done
