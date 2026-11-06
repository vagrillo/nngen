"""English caption templates for corpus networks.

A caption = lead sentence (task-specific) + I/O description + quality clause.
Slots: {io} from TaskSpec.io_desc, {quality} adjective, {score} numeric.
Template choice is seeded per sample so corpus generation is reproducible.
"""

from __future__ import annotations

import random

# Task-specific lead sentences (behavior-level description).
LEADS: dict[str, list[str]] = {
    "seek": [
        "A tiny MLP controller for a chase game: it steers a point mass to follow a smoothly moving target across a 2D arena.",
        "A small neural steering policy that chases a moving target, matching its position and velocity.",
        "A game-AI network that pursues a Lissajous-moving target with smooth acceleration commands.",
        "A pursuit controller: the network watches its own state and the target's and produces thrust toward it.",
        "A tiny policy network for target-following gameplay in a bounded square world.",
    ],
    "evade": [
        "A tiny MLP controller for an escape game: it flees a relentless pursuer on a 2D plane, never getting caught.",
        "A small neural evasion policy that keeps its distance from a chasing agent.",
        "A survival-game network that dodges a proportional-pursuit adversary with sharp acceleration commands.",
        "An evasive-maneuver controller: it reads the pursuer's state and steers away to stay alive.",
        "A tiny policy network that outmaneuvers a constant-speed chaser in a bounded arena.",
    ],
    "balance": [
        "A tiny MLP controller for a balance game: it keeps an inverted pendulum upright on a moving cart.",
        "A small neural controller that stabilizes a classic cart-pole with gentle horizontal forces.",
        "A physics-gameplay network that balances a pole on a cart, correcting tilt before it grows.",
        "An inverted-pendulum stabilizer: tiny force inputs, small angle errors, long upright times.",
        "A tiny policy network for cart-pole balancing, holding the pole near vertical for whole episodes.",
    ],
    "bounce": [
        "A tiny MLP controller for a pong-like game: it slides a paddle along the bottom wall to intercept a bouncing ball.",
        "A small neural paddle controller that tracks a bouncing ball and rackets it back up.",
        "A game-AI network that keeps a rally alive by lining up under a falling ball every time.",
        "An interception controller: it reads ball position and velocity and moves the paddle to hit it.",
        "A tiny policy network for paddle-and-ball gameplay, chaining successful hits.",
    ],
    "gait": [
        "A tiny MLP controller for procedural animation: it drives a two-joint leg so the foot traces a smooth stride ellipse.",
        "A small inverse-kinematics network that animates a leg's hip and knee through a periodic gait cycle.",
        "A walking-animation controller: given the gait phase it moves hip and knee so the foot follows an elliptical stride path.",
        "A gait oscillator network producing smooth, periodic joint-angle motion for a two-segment leg.",
        "A tiny policy network that turns gait-phase inputs into lifelike leg-joint motion.",
    ],
    "orbit": [
        "A tiny MLP controller for a space scene: it thrusts a satellite into a clean circular orbit at the requested radius.",
        "A small neural flight controller that corrects radial and tangential errors to maintain a circular orbit.",
        "An orbital-stabilization network that reads position and velocity and fires thrusters to keep a steady circle.",
        "A tiny policy network that circularizes a perturbed orbit around a planet and holds it there.",
        "A satellite station-keeping controller: minimal thrust, stable radius, constant angular speed.",
    ],
    "draw_circle": [
        "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a clean, closed circle.",
        "A small turtle-graphics network whose pen path is a neat circle of the requested size.",
        "A sketching controller that turns a pen step by step to draw a round, closed loop.",
        "A drawing-robot policy: from pen pose alone it produces the turn and step commands of a circle.",
        "A tiny policy network for procedural drawing: it inks a full circle starting from the pen's initial pose.",
    ],
    "draw_square": [
        "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a crisp square outline.",
        "A small turtle-graphics network whose pen path is a four-sided square of the requested size.",
        "A sketching controller that walks a pen around a closed square, turning sharply at each corner.",
        "A drawing-robot policy: from pen pose alone it produces the turn and step commands of a square.",
        "A tiny policy network for procedural drawing: it inks a closed square loop from a standing start.",
    ],
    "draw_triangle": [
        "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a closed equilateral triangle.",
        "A small turtle-graphics network whose pen path is a three-sided triangle of the requested size.",
        "A sketching controller that walks a pen around a triangle, turning at each vertex until the loop closes.",
        "A drawing-robot policy: from pen pose alone it produces the turn and step commands of a triangle.",
        "A tiny policy network for procedural drawing: it inks a closed triangular path.",
    ],
    "draw_star": [
        "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a five-pointed star.",
        "A small turtle-graphics network whose pen path zig-zags through the ten vertices of a star and closes it.",
        "A sketching controller that draws a classic five-pointed star with sharp turns at each point.",
        "A drawing-robot policy: from pen pose alone it produces the turn and step commands of a star polygon.",
        "A tiny policy network for procedural drawing: it inks a closed five-point star from a standing start.",
    ],
    "draw_pentagon": [
        "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a closed regular pentagon.",
        "A small turtle-graphics network whose pen path turns by a fixed 72-degree angle at each of five corners.",
        "A sketching controller that walks a pen around a five-sided polygon, turning the same amount at every vertex.",
        "A drawing-robot policy: from pen pose alone it produces the constant turn and step commands of a pentagon.",
        "A tiny policy network for procedural drawing: it inks a closed five-sided regular polygon.",
    ],
    "draw_line": [
        "A tiny MLP drawing agent: it drives a pen on a 2D canvas along a straight line of the requested length.",
        "A small turtle-graphics network whose pen path is a single straight segment, held steady without turning.",
        "A sketching controller that keeps a pen on course, drawing one clean straight stroke.",
        "A drawing-robot policy: from pen pose alone it produces near-zero turns and constant steps along a line.",
        "A tiny policy network for procedural drawing: it inks a straight horizontal stroke from a standing start.",
    ],
    "draw_arc": [
        "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a smooth half-circle arc.",
        "A small turtle-graphics network whose pen path curves steadily through a semicircle of the requested size.",
        "A sketching controller that turns the pen at a constant rate to draw a half-circle arc.",
        "A drawing-robot policy: from pen pose alone it produces the continuous turning commands of an arc.",
        "A tiny policy network for procedural drawing: it inks a constant-curvature half-circle stroke.",
    ],
    "draw_spiral": [
        "A tiny MLP drawing agent: it drives a pen on a 2D canvas to trace a smooth Archimedean spiral.",
        "A small turtle-graphics network whose pen path winds inward in a steady spiral of the requested size.",
        "A sketching controller that gradually tightens its turns to draw a two-and-a-half-turn spiral.",
        "A drawing-robot policy: from pen pose alone it produces the turn and step commands of a spiral.",
        "A tiny policy network for procedural drawing: it inks a spiraling curve from the outer rim inward.",
    ],
}

# Quality clause, keyed by percentile bucket within the task.
QUALITY_CLAUSES: dict[str, list[str]] = {
    "excellent": [
        "Rollout quality after training: excellent (score {score:.2f}).",
        "It nails the behavior: score {score:.2f} on held-out episodes.",
        "Performance: excellent, score {score:.2f}.",
    ],
    "good": [
        "Rollout quality after training: good (score {score:.2f}).",
        "It performs the behavior reliably: score {score:.2f} on held-out episodes.",
        "Performance: good, score {score:.2f}.",
    ],
    "decent": [
        "Rollout quality after training: decent (score {score:.2f}).",
        "It mostly manages the behavior, with some wobble: score {score:.2f} on held-out episodes.",
        "Performance: decent, score {score:.2f}.",
    ],
    "rough": [
        "Rollout quality after training: rough (score {score:.2f}); the behavior is only partially learned.",
        "It attempts the behavior but is inconsistent: score {score:.2f} on held-out episodes.",
        "Performance: rough, score {score:.2f}.",
    ],
}

_IO_LEAD = [
    "Inputs and outputs: {io}",
    "{io}",
    "Interface: {io}",
]


def quality_bucket(rank: float) -> str:
    """rank = percentile of the score within its task, in [0, 1]."""
    if rank >= 0.85:
        return "excellent"
    if rank >= 0.6:
        return "good"
    if rank >= 0.35:
        return "decent"
    return "rough"


def build_caption(task_name: str, io_desc: str, rank: float, score: float, seed: int) -> str:
    """Assemble one caption. Deterministic given (task_name, seed)."""
    rng = random.Random(hash((task_name, seed)) & 0xFFFF_FFFF)
    lead = rng.choice(LEADS[task_name])
    io = rng.choice(_IO_LEAD).format(io=io_desc)
    quality = rng.choice(QUALITY_CLAUSES[quality_bucket(rank)]).format(score=score)
    return f"{lead} {io} {quality}"
