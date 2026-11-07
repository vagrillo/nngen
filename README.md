# GenAIGenAI — When AI Generates Generative AI

<img width="653" height="473" alt="image" src="https://github.com/user-attachments/assets/ec9cbb96-6cc8-4140-abdc-2a6bec5a8112" />
<img width="654" height="541" alt="image" src="https://github.com/user-attachments/assets/64524543-f0e0-4504-ade3-bfe882546032" />
<img width="659" height="912" alt="image" src="https://github.com/user-attachments/assets/fcd71f86-20ce-44be-ab86-b92c37d7857a" />



**NNGen** is a generator of generative models: a diffusion model with a
text-to-image architecture that, **instead of generating images, generates a
GenAI model** — a working neural network whose weights it denoises from pure
noise, guided by a text description.

Where text-to-image systems learn the mapping *caption → pixels*, NNGen learns
*caption → network weights*: describe a behavior in English, and the model
produces a brand-new neural network that performs it. Same machinery — UNet,
v-prediction, cosine schedule, DDIM sampling, classifier-free guidance — one
level of abstraction higher: **from pictures of things, to the things
themselves.**

```
Prompt:  "A tiny MLP drawing agent: it drives a pen on a 2D canvas
          to trace a five-pointed star."
             │  diffusion sampling (DDIM, CFG)
             ▼
Output:  a NEW neural network (~1.6k parameters) that draws a star,
         exported as safetensors + rollout GIF + quality metrics
```

## How it works

Training pairs mirror the text-to-image recipe — *(image, caption)* becomes
*(network, caption)*:

- **The "image"** is a 6×32×32 *weight canvas*: the six parameter tensors of a
  fixed-size controller MLP (`8 → 32 → 32 → 8`, ~1.6k parameters), laid out
  like image channels with a validity mask, z-normalized into `[-1, 1]` — the
  same regime as pixels.
- **The caption** is an English description of the behavior ("a tiny MLP that
  drives a pen to trace a five-pointed star"), encoded by a frozen MiniLM and
  injected via cross-attention.
- **The corpus is self-generated**: thousands of controllers are trained
  through differentiable mini-environments for games and animation tasks
  (behavior cloning from analytic experts where needed), filtered by strict
  held-out rollout criteria, and captioned from templates. **No human-labeled
  data — the dataset is manufactured by compute.**

A compact text-conditioned UNet (2.3M parameters, `mvp` preset) learns to
denoise weight canvases. Sampling from a prompt yields a network that is
decoded and executed directly in its environment.

## Task families

| task | domain | behavior |
|---|---|---|
| `seek` | game | chase a moving target on a 2D arena |
| `evade` | game | escape a proportional-pursuit chaser |
| `balance` | game | stabilize a cart-pole |
| `bounce` | game | paddle intercepts a bouncing ball |
| `gait` | animation | two-joint leg IK: foot traces a stride ellipse |
| `orbit` | animation | thrust a satellite into a circular orbit |
| `draw_*` | drawing | turtle-graphics pen traces a target shape |

The drawing demo includes circle, square, triangle, pentagon, star, star6,
spiral, line and arc — while **hexagon and octagon are deliberately held out**
of the corpus to measure compositional generalization: if the model draws a
hexagon it has never seen, it learned the *concept* of a regular polygon, not
a lookup table of shapes.

## Web demo

```bash
python webapp/server.py --port 17656     # then open http://localhost:17656
```

A standard-library-only Python server (no framework, no build step) with all
URLs relative, ready to host behind a reverse proxy at any domain/sub-path.
It serves the English documentation, a live demo (prompt → generated network →
rollout GIF + weight heatmap + metrics via a background job queue) and a small
JSON API (`POST api/generate`, `GET api/job/<id>`, `GET api/status`).

## Running the pipeline

```bash
pip install -r requirements.txt

# 1. mass-train controllers through the differentiable environments
python -m dnn.corpus.train_controllers --tasks draw_circle,draw_square,draw_triangle,draw_pentagon,draw_star,draw_star6,draw_spiral,draw_line,draw_arc \
    --per-task 300 --out data/corpus_draw

# 2. cache text embeddings (one-time MiniLM download)
python -m dnn.corpus.embed_captions --corpus data/corpus_draw

# 3. train the generator (mvp preset = CPU-feasible)
python -m dnn.train_diffusion --config mvp --corpus data/corpus_draw --run runs/draw_mvp

# 4. generate a network from a prompt (CLI)
python -m dnn.generate --prompt "..." --task draw_hexagon --ckpt runs/draw_mvp/ckpt.pt --out runs/demo

# 5. quantitative eval: generated vs teacher vs random
python -m dnn.eval --ckpt runs/draw_mvp/ckpt.pt

# 6. the website
python webapp/server.py --port 17656
```

## Repository layout

```
dnn/
  config.py            fixed sizes + presets (smoke/mvp/small/base)
  controller.py        canonical MLP, weight canvas, normalization, stats
  text_encoder.py      frozen MiniLM wrapper
  tasks/               mini-environments + draw shapes, registry
  corpus/              caption templates, batch trainer (expert cloning),
                       incremental storage, embedding cache
  diffusion/           cosine schedule, UNet (cross-attn), DDIM + CFG, EMA
  train_diffusion.py   training loop (EMA, checkpoints, resume, val loss)
  generate.py          prompt → network → rollout → artifacts
  eval.py              generated vs teacher vs random benchmark
  render.py            behavior GIFs and weight heatmaps
webapp/
  server.py            NNGen website + demo API (stdlib HTTP, port 17656)
  pages/, static/      English documentation and demo UI
tests/                 canvas round-trip, tasks, diffusion, e2e pipeline
scripts/               demos and utilities
```

## Status & honest limits

- **Research prototype**: one fixed-size controller architecture, simple
  behaviors, CPU-only training (the `mvp` preset trains in a few hours on 8
  cores; a GPU multiplies corpus size and model capacity by an order of
  magnitude).
- The generator can only produce behaviors resembling its training corpus —
  and, in the best case, *interpolate* between them (that is exactly what the
  held-out hexagon/octagon experiment measures).
- Success is measured, not promised: `eval.py` reports generated networks
  against their corpus teachers and random initialization on unseen captions.

## The point

Diffusion models only ever manipulate matrices of numbers — and a weight
matrix is exactly that. Nothing in the process knows it is generating
"intelligence"; it just learns which matrices, sampled from noise, accomplish
what the caption says. That is the step this project demonstrates: **using a
generative-model technique to generate the generative models themselves.**
