"""NNGen demo server — a text-to-neural-network generator, served on the web.

Standard-library only (http.server, threading): no framework, no build step.
Every path is resolved relative to this file's directory and every URL emitted
by the pages is relative, so the site can be hosted behind a reverse proxy at
any sub-path.

Usage:
    python webapp/server.py [--port 17656] [--bind 0.0.0.0]

Endpoints:
    GET  /                     home + explanation
    GET  /docs                 documentation
    GET  /demo                 interactive demo (form + polling UI)
    POST /api/generate         {prompt, task, guidance, steps, seed} -> {job}
    GET  /api/job/<id>         job status + report when done
    GET  /api/tasks            available rollout tasks
    GET  /artifacts/<job>/<f>  generated weights PNG / behavior GIF / report
    GET  /static/<f>           stylesheet & assets
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent          # webapp/
ROOT = BASE.parent                              # repository root
PAGES = BASE / "pages"
STATIC = BASE / "static"
ARTIFACTS = ROOT / "runs" / "web_demo"          # generated jobs land here

PORT = 17656
CHECKPOINT = ROOT / "runs" / "draw_mvp" / "ckpt.pt"

# ---------------------------------------------------------------------------
# Generation worker: one job at a time (CPU-bound), results under ARTIFACTS.
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_queue: "queue.Queue[str]" = queue.Queue()
_lock = threading.Lock()


class DemoBackend:
    """Lazy singleton holding the checkpoint, diffusion sampler and encoder."""

    def __init__(self) -> None:
        self._loaded = False
        self.model = None
        self.diff = None
        self.stats = None
        self.null = None
        self.encoder = None
        self.error: str | None = None

    def load(self) -> bool:
        if self._loaded:
            return self.error is None
        self._loaded = True
        try:
            import torch

            from dnn.config import DiffusionConfig
            from dnn.diffusion.sampler import Diffusion
            from dnn.diffusion.unet import UNet
            from dnn.text_encoder import TextEncoder
            from dnn.train_diffusion import load_checkpoint

            ckpt = load_checkpoint(CHECKPOINT)
            self.model = UNet(DiffusionConfig(**ckpt["model_cfg"]))
            self.model.load_state_dict({k: v.to(torch.float32) for k, v in ckpt["ema"].items()})
            self.model.eval()
            self.diff = Diffusion(DiffusionConfig(**ckpt["model_cfg"]))
            from dnn.controller import ParamStats

            self.stats = ParamStats.from_json(ckpt["stats"])
            self.encoder = TextEncoder()
            self.null = self.encoder.encode([""])
        except Exception as e:  # missing checkpoint, missing deps, ...
            self.error = f"{type(e).__name__}: {e}"
        return self.error is None


BACKEND = DemoBackend()


def run_job(job_id: str, params: dict) -> None:
    """Generate a network from a prompt; write artifacts + report."""
    job = _jobs[job_id]
    out = ARTIFACTS / job_id
    out.mkdir(parents=True, exist_ok=True)
    try:
        if not CHECKPOINT.exists():
            raise RuntimeError(
                "No trained checkpoint found at runs/draw_mvp/ckpt.pt. "
                "Train one first: python -m dnn.train_diffusion --config mvp --corpus data/corpus_draw --run runs/draw_mvp"
            )
        if not BACKEND.load():
            raise RuntimeError(f"Failed to load the model: {BACKEND.error}")

        from dnn.generate import canvas_to_controller, generate_canvas
        from dnn.render import render_gif, render_weights
        from dnn.tasks import get_task

        canvas, stats, info = generate_canvas(
            params["prompt"],
            CHECKPOINT,
            guidance=params["guidance"],
            steps=params["steps"],
            seed=params["seed"],
            encoder=BACKEND.encoder,
        )
        net = canvas_to_controller(canvas, stats)

        task = get_task(params["task"])
        states = task.rollout(net, 1, 777_000 + params["seed"])
        score = float(task.score(states)[0])
        try:
            success = bool(task.success(states)[0])
        except NotImplementedError:
            success = None
        metrics = {k: round(v, 4) for k, v in task.report(states).items()}

        from dnn.controller import flat_to_canvas

        render_weights(flat_to_canvas(net.get_flat().unsqueeze(0))[0], out / "weights.png")
        render_gif(params["task"], task, states, out / "behavior.gif")

        job["result"] = {
            "prompt": params["prompt"],
            "task": params["task"],
            "guidance": params["guidance"],
            "steps": params["steps"],
            "seed": params["seed"],
            "score": round(score, 4),
            "success": success,
            "metrics": metrics,
            "artifacts": {
                "weights_png": f"artifacts/{job_id}/weights.png",
                "behavior_gif": f"artifacts/{job_id}/behavior.gif",
                "report": f"artifacts/{job_id}/report.json",
            },
        }
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()


def worker_loop() -> None:
    while True:
        job_id = _queue.get()
        job = _jobs[job_id]
        job["status"] = "running"
        try:
            run_job(job_id, job["params"])
        finally:
            pass


threading.Thread(target=worker_loop, daemon=True).start()

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
TASK_CATALOG = [
    ("draw_circle", "circle"), ("draw_square", "square"), ("draw_triangle", "triangle"),
    ("draw_pentagon", "pentagon"), ("draw_star", "star"), ("draw_spiral", "spiral"),
    ("draw_line", "line"), ("draw_arc", "arc"),
    ("draw_hexagon", "hexagon (held-out!)"), ("draw_octagon", "octagon (held-out!)"),
    ("seek", "seek (chase)"), ("gait", "gait (leg IK)"), ("orbit", "orbit (satellite)"),
]

PROMPT_PRESETS = {
    "draw_circle": "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a clean, closed circle.",
    "draw_pentagon": "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a closed regular pentagon, turning by a fixed angle at each of five corners.",
    "draw_spiral": "A small turtle-graphics network whose pen path winds inward in a steady spiral of the requested size.",
    "draw_line": "A tiny MLP drawing agent: it drives a pen along a straight line, holding its heading steady.",
    "draw_hexagon": "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a closed regular hexagon, turning by a fixed 60-degree angle at each of six corners.",
    "draw_octagon": "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a closed regular octagon, turning by a fixed 45-degree angle at each of eight corners.",
    "seek": "A tiny MLP controller for a chase game: it steers a point mass to follow a smoothly moving target across a 2D arena.",
    "gait": "A tiny MLP controller for procedural animation: it drives a two-joint leg so the foot traces a smooth stride ellipse.",
    "orbit": "A tiny MLP controller for a space scene: it thrusts a satellite into a clean circular orbit at the requested radius.",
}


def page(name: str) -> str:
    html = (PAGES / name).read_text()
    header = (PAGES / "_header.html").read_text()
    footer = (PAGES / "_footer.html").read_text()
    return html.replace("{{header}}", header).replace("{{footer}}", footer)


class Handler(BaseHTTPRequestHandler):
    server_version = "NNGen/0.1"

    # -- helpers ------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, code: int, html: str) -> None:
        self._send(code, html.encode(), "text/html; charset=utf-8")

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _file(self, path: Path, ctype: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self._html(404, "<h1>404</h1>")
            return
        self._send(200, body, ctype)

    def log_message(self, fmt, *args):  # quieter logs
        pass

    # -- routing ------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].strip("/")
        try:
            if path == "":
                self._html(200, page("index.html"))
            elif path == "docs":
                self._html(200, page("docs.html"))
            elif path == "demo":
                self._html(200, page("demo.html"))
            elif path == "api/tasks":
                self._json(200, {"tasks": [{"id": t, "label": l} for t, l in TASK_CATALOG]})
            elif path == "api/status":
                ready = CHECKPOINT.exists() and BACKEND.load()
                self._json(200, {"ready": ready, "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
                                 "error": None if ready else (BACKEND.error or "checkpoint missing")})
            elif path.startswith("api/job/"):
                job_id = path.split("/")[-1]
                job = _jobs.get(job_id)
                if job is None:
                    self._json(404, {"error": "unknown job"})
                else:
                    self._json(200, {k: job[k] for k in ("status", "result", "error") if k in job})
            elif path.startswith("artifacts/"):
                self._serve_under(ARTIFACTS, path[len("artifacts/"):])
            elif path.startswith("static/"):
                self._serve_under(STATIC, path[len("static/"):])
            else:
                self._html(404, page("404.html") if (PAGES / "404.html").exists() else "<h1>404 not found</h1>")
        except BrokenPipeError:
            pass
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.strip("/")
        try:
            if path == "api/generate":
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    self._json(400, {"error": "prompt is required"})
                    return
                params = {
                    "prompt": prompt[:500],
                    "task": str(payload.get("task", "draw_hexagon")),
                    "guidance": float(payload.get("guidance", 4.0)),
                    "steps": max(5, min(60, int(payload.get("steps", 30)))),
                    "seed": int(payload.get("seed", 42)),
                }
                if params["task"] not in {t for t, _ in TASK_CATALOG}:
                    self._json(400, {"error": f"unknown task '{params['task']}'"})
                    return
                job_id = uuid.uuid4().hex[:12]
                with _lock:
                    _jobs[job_id] = {"status": "pending", "params": params, "result": None, "error": None}
                _queue.put(job_id)
                self._json(200, {"job": job_id})
            else:
                self._json(404, {"error": "unknown endpoint"})
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    # -- safe file serving (no path traversal) ------------------------------
    def _serve_under(self, base: Path, rel: str) -> None:
        target = (base / rel).resolve()
        if not str(target).startswith(str(base.resolve())):
            self._html(403, "<h1>403</h1>")
            return
        ctype = {
            ".png": "image/png", ".gif": "image/gif", ".json": "application/json",
            ".css": "text/css", ".js": "application/javascript", ".svg": "image/svg+xml",
        }.get(target.suffix, "application/octet-stream")
        self._file(target, ctype)


def main() -> None:
    global PORT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    PORT = args.port
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"NNGen demo server listening on http://{args.bind}:{args.port} (all paths relative to ./)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
