"""Emits the chaos-swarm + vortex fragment for docs/banner.html.

The scene: a warp pipe at the centre, and a vortex of memory candidates funnelling into it.
One neuron is caught in the stream (the right one). The brightest one in the swarm is a decoy
whose trail breaks before the mouth.

Deterministic: fixed seed, so a re-run reproduces the same swarm byte for byte. Paste the output
between the CHAOS-BEGIN / CHAOS-END markers in banner.html.

    python docs/make_banner_graph.py > docs/_chaos.svg
"""
from __future__ import annotations

import math
import random

SEED = 20260804
CX, MOUTH_Y = 300.0, 440.0
R_MOUTH, R_TOP = 56.0, 190.0
TOP_Y = 100.0
KY = 0.30            # vertical squash: the funnel is seen from slightly above
TURNS = 4.4          # radians of swirl between the mouth and the rim

CYAN = ["#67e8f9", "#22d3ee", "#5eead4", "#38bdf8"]
VIOLET = ["#a78bfa", "#c4b5fd", "#8b5cf6"]


def ring_y(t: float) -> float:
    return MOUTH_Y - (MOUTH_Y - TOP_Y) * (t ** 1.12)


def ring_r(t: float) -> float:
    return R_MOUTH + (R_TOP - R_MOUTH) * t


def funnel(t: float, th0: float, jr: float = 1.0, jy: float = 0.0) -> tuple[float, float]:
    th = th0 + TURNS * t
    r = ring_r(t) * jr
    return CX + r * math.cos(th), ring_y(t) + r * KY * math.sin(th) + jy


def spiral(th0: float, t_hi: float, t_lo: float, n: int = 96) -> str:
    pts = []
    for i in range(n + 1):
        t = t_hi + (t_lo - t_hi) * i / n
        x, y = funnel(t, th0)
        pts.append(f"{x:.1f},{y:.1f}")
    return "M " + " L ".join(pts)


def main() -> None:
    rng = random.Random(SEED)
    out: list[str] = []

    # ---- vortex arcs: the funnel, read as motion ----------------------------------------
    out.append('<g fill="none" stroke="#38bdf8" stroke-linecap="round">')
    for th0, op, w in ((0.0, .13, 1.0), (1.9, .17, 1.1), (3.4, .12, .95), (5.0, .20, 1.15)):
        out.append(f'  <path d="{spiral(th0, 1.0, 0.02)}" stroke-opacity="{op}" stroke-width="{w}"/>')
    out.append("</g>")

    # ---- the swarm ---------------------------------------------------------------------
    nodes: list[tuple[float, float, float, str, float]] = []
    for _ in range(132):
        t = rng.uniform(0.05, 1.0) ** 0.85
        th = rng.uniform(0, 2 * math.pi)
        x, y = funnel(t, th, jr=rng.uniform(0.70, 1.08), jy=rng.uniform(-11, 11))
        base = rng.uniform(0.9, 2.5) * (1.28 - 0.58 * t)
        col = rng.choice(CYAN) if rng.random() < 0.66 else (
            "#ffffff" if rng.random() < 0.22 else rng.choice(VIOLET))
        op = min(0.72, 0.20 + 0.52 * (1 - t) + rng.uniform(-.06, .10))
        nodes.append((x, y, base, col, op))

    # a handful of larger ones, so the swarm has a foreground
    big: list[tuple[float, float, float, str, float]] = []
    for _ in range(22):
        t = rng.uniform(0.10, 0.95) ** 0.8
        th = rng.uniform(0, 2 * math.pi)
        x, y = funnel(t, th, jr=rng.uniform(0.76, 1.04), jy=rng.uniform(-8, 8))
        r = rng.uniform(3.4, 6.4) * (1.20 - 0.40 * t)
        col = rng.choice(CYAN) if rng.random() < 0.62 else rng.choice(VIOLET)
        big.append((x, y, r, col, min(0.85, 0.42 + 0.40 * (1 - t))))

    # ---- web: near neighbours only, so it reads as structure not spaghetti --------------
    alln = nodes + big
    edges = []
    for i in range(len(alln)):
        for j in range(i + 1, len(alln)):
            dx, dy = alln[i][0] - alln[j][0], alln[i][1] - alln[j][1]
            d = math.hypot(dx, dy / KY * 0.42)
            if d < 46:
                edges.append((alln[i], alln[j], d))
    rng.shuffle(edges)
    edges = edges[:118]
    out.append('<g fill="none" stroke="#38bdf8" stroke-width=".7" stroke-linecap="round">')
    for a, b, d in edges:
        op = 0.20 * (1 - d / 46) + 0.05
        out.append(f'  <path d="M{a[0]:.1f},{a[1]:.1f} L{b[0]:.1f},{b[1]:.1f}" stroke-opacity="{op:.2f}"/>')
    out.append("</g>")

    out.append('<g filter="url(#bloom)">')
    for x, y, r, col, op in big:
        out.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{r * 1.5:.1f}" fill="{col}" fill-opacity="{op * .7:.2f}"/>')
    out.append("</g>")

    out.append("<g>")
    for x, y, r, col, op in nodes:
        out.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}" fill="{col}" fill-opacity="{op:.2f}"/>')
    for x, y, r, col, op in big:
        out.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{col}" fill-opacity="{op:.2f}"/>')
    out.append("</g>")

    # ---- the one it collects, and the decoy it refuses ----------------------------------
    TH_HIT, T_HIT = 2.55, 0.34
    TH_DEC, T_DEC = 1.20, 0.86
    hx, hy = funnel(T_HIT, TH_HIT)
    dx_, dy_ = funnel(T_DEC, TH_DEC)
    bx, by = funnel(0.50, TH_DEC)          # where the decoy's trail breaks off

    out.append(f'<!-- hit {hx:.1f},{hy:.1f}  decoy {dx_:.1f},{dy_:.1f}  break {bx:.1f},{by:.1f} -->')
    out.append(f'<path d="{spiral(TH_HIT, 0.48, 0.012)}" fill="none" stroke="url(#trail)" '
               f'stroke-width="2.6" stroke-linecap="round"/>')
    out.append(f'<path d="{spiral(TH_DEC, 0.86, 0.50)}" fill="none" stroke="#f5a524" '
               f'stroke-opacity=".55" stroke-width="1.8" stroke-dasharray="6 7" stroke-linecap="round"/>')
    out.append(f'<g stroke="#f5a524" stroke-opacity=".85" stroke-width="2" stroke-linecap="round">'
               f'<path d="M{bx - 6:.1f},{by - 6:.1f} L{bx + 6:.1f},{by + 6:.1f}"/>'
               f'<path d="M{bx + 6:.1f},{by - 6:.1f} L{bx - 6:.1f},{by + 6:.1f}"/></g>')

    print("\n".join(out))
    print(f"<!-- POS hit={hx:.1f},{hy:.1f} decoy={dx_:.1f},{dy_:.1f} break={bx:.1f},{by:.1f} -->")


if __name__ == "__main__":
    main()
