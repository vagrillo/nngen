# diffusionNN — text-to-network diffusion model

A diffusion model with a **text2image architecture** (UNet + text cross-attention +
classifier-free guidance + v-prediction DDIM) where the "image" being denoised is
**the weight tensor of a neural network** instead of a picture.

Training pairs mirror the text2image recipe — `(image, caption)` becomes
`(network, caption)`:

- the **image** is the 6×32×32 weight *canvas* of a fixed-size controller MLP
  (`8 → 32 → 32 → 8`, ~1.6k parameters), like a fixed-resolution image;
- the **caption** is an English description of what the network does
  ("a tiny MLP controller that drives a pen to trace a five-pointed star");
- the **corpus** is synthesized: thousands of controllers are trained through
  differentiable mini-environments for **games and animation** tasks, filtered
  by held-out rollout success, and captioned from templates.

Sampling from a prompt yields a **brand-new network** that is loaded and executed
directly in the target environment. A model that generates models.

## Task families (all drive the same canonical MLP)

| task | domain | behavior |
|---|---|---|
| `seek` | game | chase a moving target on a 2D arena |
| `evade` | game | escape a proportional-pursuit chaser |
| `balance` | game | stabilize a cart-pole |
| `bounce` | game | paddle intercepts a bouncing ball (pong-like) |
| `gait` | animation | two-joint leg IK: foot traces a stride ellipse |
| `orbit` | animation/space | thrust a satellite into a circular orbit |
| `draw_circle` / `draw_square` / `draw_triangle` / `draw_star` / `draw_spiral` | drawing | turtle-graphics pen traces the shape |

## Pipeline

```bash
pip install -r requirements.txt

# 1. Mass-train controllers through the differentiable environments (~1-3 h CPU)
python -m dnn.corpus.train_controllers --tasks all --per-task 600 --out data/corpus

# 2. Cache MiniLM text embeddings for the captions (one-time download)
python -m dnn.corpus.embed_captions --corpus data/corpus

# 3. Train the diffusion model (smoke ~30 min; small ~4-6 h; base overnight, CPU)
python -m dnn.train_diffusion --config smoke --corpus data/corpus --run runs/smoke
python -m dnn.train_diffusion --config small --corpus data/corpus --run runs/small

# 4. Generate a NEW network from a text prompt and watch it act
python -m dnn.generate \
  --prompt "A tiny MLP drawing agent that drives a pen on a canvas to trace a five-pointed star" \
  --task draw_star --ckpt runs/small/ckpt.pt --out runs/demo_star

# 5. Quantitative eval: generated nets vs their corpus teachers vs random, on
#    validation captions (unseen during training)
python -m dnn.eval --ckpt runs/small/ckpt.pt --per-task 8
```

`generate.py` writes: `generated_net.safetensors` (the new model's weights),
`weights.png` (weight-canvas heatmap), `behavior.gif` (the network acting in
its environment) and `report.json` (rollout score/success).

## How the "image" works

The canonical controller has 6 parameter tensors. Each is placed on a 32×32
channel (matrices top-left, biases in the first column); a static validity mask
(marked where real weights live) is concatenated as extra UNet input channels.
Weights are z-normalized per tensor using corpus statistics, clamped to ±4σ →
`[-1, 1]`. Training loss and decoding are masked to valid entries, so padding
stays exactly zero. `dnn/controller.py` owns this mapping; round-trip is tested.

## Design decisions

- **v-prediction** + cosine schedule + DDIM (50 steps): stable, CPU-friendly.
- **Classifier-free guidance** (caption dropout 10%, guidance ≈ 3-5) for prompt
  fidelity, exactly as in text2image.
- **Frozen MiniLM** (`all-MiniLM-L6-v2`, 384-dim) encodes captions; token-level
  embeddings feed the UNet cross-attention.
- **torch.compile** traces each task's rollout+loss+backward once (~10× on CPU);
  tasks train in parallel worker processes.
- Environment instances are **resampled every optimizer step** (domain
  randomization) and networks are filtered on a **held-out episode seed**, so
  the corpus contains robust behavior, not memorized episodes.

## Repository layout

```
dnn/
  config.py            fixed sizes + presets (smoke/small/base)
  controller.py        canonical MLP, weight canvas, normalization, stats
  text_encoder.py      frozen MiniLM wrapper
  tasks/               base + 6 families + draw shapes, registry
  corpus/              caption templates, batch trainer, storage, embedding cache
  diffusion/           cosine schedule, UNet (cross-attn), DDIM sampler + CFG, EMA
  train_diffusion.py   training loop (EMA, checkpoints, resume, val loss)
  generate.py          prompt → network → rollout → artifacts
  eval.py              generated vs teacher vs random success table
  render.py            behavior GIFs and weight heatmaps
tests/                 canvas round-trip, tasks, diffusion, pipeline e2e
```

## Status / notes

- CPU-only development environment; `smoke` preset validates the full loop in
  ~30 minutes, `small` is the primary configuration.
- The generated-network quality gate is `eval.py`: on validation captions the
  generated nets should score close to (typically below) their corpus teachers
  and far above random initialization. See `runs/<name>/eval.md`.
