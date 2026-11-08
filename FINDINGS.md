# FINDINGS — an honest experiment log

This document records what currently works, what does not, and why. NNGen is
an exploration: negative results are part of it, and we publish them with the
same care as the positives.

## The question

> If I pass a specific text prompt to the trained diffusion model, does the
> generated network — when executed — actually perform the behavior the prompt
> describes? Do we have at least one verified end-to-end success case?

## Current answer (as of this writing): **not yet**

We ran a systematic **success hunt**: every task family trained into the
corpus, 4 seeds each, 50-step DDIM sampling, guidance 4.0, judged by the
strict automatic criteria (point accuracy, full-curve coverage, closure,
pace on a held-out environment episode).

| task | result | best score | closest failure mode |
|---|---|---|---|
| `draw_line` | 0/4 | 0.030 | traces part of the line, stops early |
| `draw_arc` | 0/4 | 0.023 | partial arc, endpoint misses |
| `draw_spiral` | 0/4 | **0.195** | most of the spiral, slight drift |
| `draw_circle` | 0/4 | 0.034 | small arc, rest uncovered |
| `draw_pentagon` | 0/4 | 0.143 | partial trace |
| `draw_square` | 0/4 | 0.000 | single S-stroke |
| `draw_star` / `draw_star6` | 0/4 | 0.000 | degenerate |
| **total** | **0/28** | | |

A consistent signature across all failures: **curve_err is decent
(0.02–0.07 — the pen draws near the right curve locally) but coverage_err is
high (0.17–0.68) and endpoint/closure fails — the generated networks trace
part of the target curve and do not complete it.** The behaviors are
recognizably "attempting the shape", not garbage — but not successes either.

## What we ruled out (measured, not guessed)

| hypothesis | test | verdict |
|---|---|---|
| sampling parameters | sweep: DDIM steps 10/30/50/200/400 × guidance 1–8 × seeds | fails everywhere |
| EMA weights lagging | EMA vs raw final weights | identical results |
| classifier-free guidance | guidance 1.0 (off), 2.0, 3.0 | sampled canvases statistically identical → text conditioning has ~no effect yet |
| long sampling | 200/400 DDIM steps | no improvement |

## Root cause (measured)

- **Sample distribution mismatch**: per-channel std of sampled canvases is
  3–30× the corpus distribution (e.g. bias channels 0.61–0.67 vs 0.02–0.04).
  The model does not yet converge samples onto the tight weight manifold.
- **Overfitting ceiling**: extending training 1500 → 6500 steps made
  validation loss *worse* (0.2238 → 0.2560, monotonic) while training loss
  fell — the 2.5k-network corpus is saturated by the 2.3M-parameter model.
- **Weak conditioning**: generated canvases are nearly independent of the
  caption at this stage.

## What unambiguously works (and is verified)

- The **corpus**: 2,916 controller networks, each one verified by the strict
  held-out criteria (they genuinely trace their curves / perform their tasks).
  The full self-generation pipeline (differentiable environments, batch
  expert cloning, quality filtering, template captions) is reproducible from
  the repository.
- The **weight-canvas round trip**, the **diffusion training/sampling stack**
  (v-prediction, DDIM, CFG), the **web demo**, and the **evaluation harness** —
  all tested.
- An earlier model release appeared to pass hexagon/octagon; that result was
  later **retracted** when the success criteria were hardened (see commit
  history: coverage criterion). Honesty over vanity.

## The path to the first success case (in progress)

1. **Dataset scaling 3×**: a ~9.7k-network corpus (9 shapes × ~1080) is being
   generated (the expert-cloning recipe makes this fast). Overfitting should
   arrive correspondingly later.
2. **Retrain with best-validation checkpointing** (small corpora overfit
   fast; keep the best generalizing weights, not the last ones).
3. **Re-run this exact success hunt** (same script, same criteria) — the
   first shape with a 4/4 or even 2/4 pass becomes the documented end-to-end
   success case, with its GIF and report published here.
4. If dataset scaling stalls, the declared next steps are a larger model
   with the larger corpus, and GPU acceleration (a L4 makes the whole loop
   10–20× faster).

## One-line summary

The pipeline is real and verified end-to-end up to the last step; the
generative step itself does not yet produce working networks at the strict
bar, and the measured reason is dataset scale — which is exactly the
hypothesis currently being tested (3× corpus, results to be published here).
