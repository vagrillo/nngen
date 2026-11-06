#!/usr/bin/env bash
# MVP demo: text -> drawing network.
#
# Shapes 1-8 are in the training corpus; hexagon and octagon are HELD OUT —
# the model has never seen a caption describing them nor their weights.
# Success there = compositional generalization in weight space.
set -e
cd "$(dirname "$0")/.."

CKPT=runs/draw_mvp/ckpt.pt
OUT=runs/demo_draw

# Seen shapes (should work)
python -m dnn.generate --prompt "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a clean, closed circle." --task draw_circle  --ckpt "$CKPT" --out "$OUT/circle"
python -m dnn.generate --prompt "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a closed regular pentagon." --task draw_pentagon --ckpt "$CKPT" --out "$OUT/pentagon"
python -m dnn.generate --prompt "A small turtle-graphics network whose pen path winds inward in a steady spiral." --task draw_spiral --ckpt "$CKPT" --out "$OUT/spiral"

# Generalization test (held-out shapes)
python -m dnn.generate --prompt "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a closed regular hexagon, turning by a fixed 60-degree angle at each of six corners." --task draw_hexagon --ckpt "$CKPT" --out "$OUT/hexagon"
python -m dnn.generate --prompt "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a closed regular octagon, turning by a fixed 45-degree angle at each of eight corners." --task draw_octagon --ckpt "$CKPT" --out "$OUT/octagon"

echo
echo "=== Summary ==="
for f in "$OUT"/*/report.json; do
    python - "$f" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
name = r["task"]
print(f"{name:14s} curve_err={r['metrics'].get('curve_err', -1):.3f} "
      f"closure={r['metrics'].get('closure_err', r['metrics'].get('endpoint_err', -1)):.3f} "
      f"len_ratio={r['metrics'].get('length_ratio', -1):.2f} score={r['score']:.2f}")
EOF
done
echo "GIFs and weight heatmaps in $OUT/<shape>/"
