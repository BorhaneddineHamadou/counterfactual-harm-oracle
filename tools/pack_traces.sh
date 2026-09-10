#!/bin/bash
# Pack the raw simulation traces of one subject into a tarball for release.
#
# The traces are the campaign stores behind data/<subject>/ (~380 MB for
# openpilot, ~630 MB for TransFuser, compressed npz). They are distributed
# as release assets rather than in git; see docs/TRACES.md.
#
#   bash tools/pack_traces.sh openpilot   [DEST_DIR]
#   bash tools/pack_traces.sh transfuser  [DEST_DIR]
#
# Layout inside the tarball:  <subject>/traces/{nominal,tier1,rq2_a05,rq2_a20}/*.npz
set -euo pipefail
SUBJECT=${1:?openpilot|transfuser}
DEST=${2:-.}
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC="$HERE/data/$SUBJECT/traces"
[ -d "$SRC" ] || { echo "no traces under $SRC"; exit 1; }
OUT="$DEST/counterfactual-harm-oracle-traces-$SUBJECT.tar"
# dereference symlinks so a linked store is packed as files
tar --dereference -cf "$OUT" -C "$HERE/data" "$SUBJECT/traces"
ls -la "$OUT"
sha256sum "$OUT"
