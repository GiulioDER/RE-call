// RE-call memory graph: every memo a point, every declared link a line.
// Plain script, no dependencies. Memo text only ever reaches the page through textContent.
(function () {
  "use strict";

  const canvas = document.getElementById("chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const frame = document.getElementById("chart-frame");
  const tip = document.getElementById("tip");
  const panel = document.getElementById("panel");
  const countsEl = document.getElementById("counts");
  const search = document.getElementById("search");
  const toggles = {
    link: document.getElementById("show-link"),
    supersedes: document.getElementById("show-supersedes"),
    pending: document.getElementById("show-pending"),
    hubs: document.getElementById("show-hubs"),
  };

  let data = null;
  let nodes = [];
  let byId = new Map();
  let edges = [];
  let active = [];
  let activeEdges = [];
  let neighbours = new Map();
  let view = { k: 1, x: 0, y: 0 };
  let hover = null;
  let selected = null;
  let matches = null;
  let alpha = 0;
  let dirty = true;
  let palette = {};

  function readPalette() {
    const css = getComputedStyle(document.documentElement);
    const v = (name) => css.getPropertyValue(name).trim();
    palette = {
      ground: v("--ground"),
      current: v("--node-current"),
      superseded: v("--node-superseded"),
      expired: v("--node-expired"),
      pending: v("--node-pending"),
      link: v("--edge-link"),
      supersedes: v("--edge-supersedes"),
      pendingEdge: v("--edge-pending"),
      select: v("--signal"),
    };
  }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function hash(text) {
    let h = 2166136261;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0) / 4294967295;
  }

  function radius(n) {
    return Math.min(9, 1.8 + Math.sqrt(n.degree) * 0.85);
  }

  // ------------------------------------------------------------ layout

  function rebuild() {
    const shown = new Set(["link", "supersedes", "pending"].filter((k) => toggles[k].checked));
    active = nodes.filter((n) => toggles.hubs.checked || !n.hub);
    const live = new Set(active.map((n) => n.id));
    activeEdges = edges.filter((e) => shown.has(e.kind) && live.has(e.source) && live.has(e.target));
    neighbours = new Map(active.map((n) => [n.id, new Set()]));
    for (const e of activeEdges) {
      neighbours.get(e.source).add(e.target);
      neighbours.get(e.target).add(e.source);
    }
    // A stable starting picture: the same memo starts in the same place on every load.
    const golden = Math.PI * (3 - Math.sqrt(5));
    active.forEach((n, i) => {
      if (n.x === undefined) {
        const r = 14 * Math.sqrt(i + 0.5);
        const a = i * golden + hash(n.id) * 0.6;
        n.x = r * Math.cos(a);
        n.y = r * Math.sin(a);
        n.vx = 0;
        n.vy = 0;
      }
    });
    alpha = 1;
    updateCounts();
    requestAnimationFrame(loop);
  }

  function tick() {
    const cell = 110;
    const grid = new Map();
    for (const n of active) {
      const key = Math.floor(n.x / cell) + "," + Math.floor(n.y / cell);
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(n);
    }
    for (const n of active) {
      const gx = Math.floor(n.x / cell);
      const gy = Math.floor(n.y / cell);
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const bucket = grid.get(gx + dx + "," + (gy + dy));
          if (!bucket) continue;
          for (const m of bucket) {
            if (m === n) continue;
            let x = n.x - m.x;
            let y = n.y - m.y;
            let d2 = x * x + y * y;
            if (d2 === 0) {
              x = hash(n.id) - 0.5;
              y = hash(m.id) - 0.5;
              d2 = 0.01;
            }
            if (d2 > cell * cell) continue;
            // Clamped at short range: two memos landing on one spot must not fling each other out.
            const f = (240 * alpha) / Math.max(d2, 49);
            n.vx += x * f;
            n.vy += y * f;
          }
        }
      }
    }
    for (const e of activeEdges) {
      const a = byId.get(e.source);
      const b = byId.get(e.target);
      const x = b.x - a.x;
      const y = b.y - a.y;
      const d = Math.sqrt(x * x + y * y) || 1;
      const rest = e.kind === "supersedes" ? 28 : 64;
      const s = ((d - rest) / d) * (e.kind === "supersedes" ? 0.05 : 0.008) * alpha;
      a.vx += x * s;
      a.vy += y * s;
      b.vx -= x * s;
      b.vy -= y * s;
    }
    for (const n of active) {
      n.vx -= n.x * 0.006 * alpha;
      n.vy -= n.y * 0.006 * alpha;
      n.vx *= 0.8;
      n.vy *= 0.8;
      const speed = Math.hypot(n.vx, n.vy);
      if (speed > 14) { n.vx *= 14 / speed; n.vy *= 14 / speed; }
      n.x += n.vx;
      n.y += n.vy;
    }
    alpha *= 0.985;
  }

  function fit() {
    if (!active.length) return;
    // Fit the body of the picture, not its strays: the 2nd to 98th percentile on each axis.
    const xs = active.map((n) => n.x).sort((a, b) => a - b);
    const ys = active.map((n) => n.y).sort((a, b) => a - b);
    const lo = Math.floor(xs.length * 0.02), hi = Math.max(lo, Math.ceil(xs.length * 0.98) - 1);
    const minX = xs[lo], maxX = xs[hi], minY = ys[lo], maxY = ys[hi];
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const k = Math.min(w / (maxX - minX + 60), h / (maxY - minY + 60), 4);
    view = { k, x: w / 2 - k * (minX + maxX) / 2, y: h / 2 - k * (minY + maxY) / 2 };
  }

  function loop() {
    if (alpha > 0.004) {
      for (let i = 0; i < 3; i++) tick();
      if (!selected) fit();
      dirty = true;
      requestAnimationFrame(loop);
    }
    if (dirty) draw();
  }

  // ------------------------------------------------------------ drawing

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    dirty = true;
    draw();
  }

  function focusSet() {
    if (selected) return new Set([selected.id, ...neighbours.get(selected.id) || []]);
    if (hover) return new Set([hover.id, ...neighbours.get(hover.id) || []]);
    return matches;
  }

  function draw() {
    dirty = false;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(view.x, view.y);
    ctx.scale(view.k, view.k);
    const focus = focusSet();
    const line = 1 / view.k;

    for (const e of activeEdges) {
      const a = byId.get(e.source), b = byId.get(e.target);
      const lit = !focus || (focus.has(a.id) && focus.has(b.id));
      ctx.beginPath();
      ctx.setLineDash(e.kind === "pending" ? [5 * line, 4 * line] : []);
      if (e.kind === "link") {
        ctx.strokeStyle = palette.link;
        ctx.globalAlpha = lit ? (focus ? 0.8 : 0.26) : 0.035;
        ctx.lineWidth = 0.75 * line;
      } else if (e.kind === "supersedes") {
        ctx.strokeStyle = palette.supersedes;
        ctx.globalAlpha = lit ? 0.9 : 0.12;
        ctx.lineWidth = 1.4 * line;
      } else {
        ctx.strokeStyle = palette.pendingEdge;
        ctx.globalAlpha = lit ? 0.95 : 0.2;
        ctx.lineWidth = 1.3 * line;
      }
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      if (e.kind !== "link" && lit) arrow(a, b, line);
    }
    ctx.setLineDash([]);

    for (const n of active) {
      const lit = !focus || focus.has(n.id);
      const r = radius(n);
      ctx.globalAlpha = lit ? 1 : 0.14;
      if (n.state === "pending") {
        ctx.beginPath();
        ctx.fillStyle = palette.pending;
        ctx.globalAlpha = lit ? 0.18 : 0.04;
        ctx.arc(n.x, n.y, r * 2.6, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = lit ? 1 : 0.14;
      }
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      if (n.state === "superseded") {
        ctx.strokeStyle = palette.superseded;
        ctx.lineWidth = 1.3 * line;
        ctx.stroke();
      } else {
        ctx.fillStyle = palette[n.state] || palette.current;
        ctx.fill();
      }
      if (n === selected || n === hover) {
        ctx.beginPath();
        ctx.strokeStyle = palette.select;
        ctx.globalAlpha = 1;
        ctx.lineWidth = 1.6 * line;
        ctx.arc(n.x, n.y, r + 3.5 * line, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.restore();
    ctx.globalAlpha = 1;
  }

  function arrow(a, b, line) {
    const angle = Math.atan2(b.y - a.y, b.x - a.x);
    const r = radius(b) + 2 * line;
    const tipX = b.x - Math.cos(angle) * r, tipY = b.y - Math.sin(angle) * r;
    const size = 6 * line;
    ctx.beginPath();
    ctx.setLineDash([]);
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - size * Math.cos(angle - 0.45), tipY - size * Math.sin(angle - 0.45));
    ctx.lineTo(tipX - size * Math.cos(angle + 0.45), tipY - size * Math.sin(angle + 0.45));
    ctx.closePath();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fill();
  }

  // ------------------------------------------------------------ interaction

  function at(px, py) {
    const x = (px - view.x) / view.k, y = (py - view.y) / view.k;
    let best = null, bestD = Infinity;
    for (const n of active) {
      const d = (n.x - x) ** 2 + (n.y - y) ** 2;
      const reach = (radius(n) + 5 / view.k) ** 2;
      if (d < reach && d < bestD) { best = n; bestD = d; }
    }
    return best;
  }

  let drag = null;
  canvas.addEventListener("pointerdown", (ev) => {
    drag = { x: ev.clientX, y: ev.clientY, vx: view.x, vy: view.y, moved: false };
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener("pointermove", (ev) => {
    const box = canvas.getBoundingClientRect();
    if (drag) {
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      view.x = drag.vx + dx;
      view.y = drag.vy + dy;
      dirty = true;
      draw();
      return;
    }
    const found = at(ev.clientX - box.left, ev.clientY - box.top);
    if (found !== hover) {
      hover = found;
      canvas.style.cursor = found ? "pointer" : "grab";
      dirty = true;
      draw();
    }
    showTip(found, ev.clientX - box.left, ev.clientY - box.top);
  });
  canvas.addEventListener("pointerup", (ev) => {
    const box = canvas.getBoundingClientRect();
    if (drag && !drag.moved) select(at(ev.clientX - box.left, ev.clientY - box.top));
    drag = null;
  });
  canvas.addEventListener("pointerleave", () => {
    hover = null;
    tip.hidden = true;
    dirty = true;
    draw();
  });
  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const box = canvas.getBoundingClientRect();
    const px = ev.clientX - box.left, py = ev.clientY - box.top;
    const k = Math.max(0.15, Math.min(12, view.k * Math.exp(-ev.deltaY * 0.0015)));
    view.x = px - ((px - view.x) / view.k) * k;
    view.y = py - ((py - view.y) / view.k) * k;
    view.k = k;
    dirty = true;
    draw();
  }, { passive: false });

  const STATE_LABEL = { current: "current", superseded: "superseded", expired: "expired", pending: "pending review" };

  function showTip(n, x, y) {
    if (!n) { tip.hidden = true; return; }
    tip.replaceChildren(el("strong", "", n.title), el("span", "tip-meta", STATE_LABEL[n.state] + " · " + n.degree + " connections"));
    tip.hidden = false;
    const w = frame.clientWidth;
    tip.style.left = Math.min(x + 14, w - tip.offsetWidth - 8) + "px";
    tip.style.top = y + 14 + "px";
  }

  function select(n) {
    selected = n;
    dirty = true;
    draw();
    renderPanel();
  }

  function renderPanel() {
    if (!selected) {
      panel.replaceChildren(el("p", "panel-empty", "Select a point to read the memo and follow its links."));
      return;
    }
    const n = selected;
    const head = el("div", "panel-head");
    head.append(el("span", "state state-" + n.state, STATE_LABEL[n.state]));
    if (n.type) head.append(el("span", "chip", n.type));
    const close = el("button", "close", "×");
    close.type = "button";
    close.setAttribute("aria-label", "Close");
    close.addEventListener("click", () => select(null));
    head.append(close);
    const parts = [head, el("h2", "", n.title), el("code", "path", n.id)];
    if (n.description) parts.push(el("p", "desc", n.description));
    if (n.claim) {
      const review = el("a", "review-link", "Review the pending claim →");
      review.href = "/review?claim=" + encodeURIComponent(n.claim);
      parts.push(review);
    }
    const groups = [
      ["Supersedes", (e) => e.kind === "supersedes" && e.source === n.id, "target"],
      ["Superseded by", (e) => e.kind === "supersedes" && e.target === n.id, "source"],
      ["Would be replaced by (pending)", (e) => e.kind === "pending" && e.target === n.id, "source"],
      ["Would replace (pending)", (e) => e.kind === "pending" && e.source === n.id, "target"],
      ["Links to", (e) => e.kind === "link" && e.source === n.id, "target"],
      ["Linked from", (e) => e.kind === "link" && e.target === n.id, "source"],
    ];
    for (const [label, test, end] of groups) {
      const items = edges.filter(test).map((e) => byId.get(e[end])).filter(Boolean);
      if (!items.length) continue;
      const section = el("section", "links");
      section.append(el("h3", "", label + " · " + items.length));
      const list = el("ul");
      for (const other of items.slice(0, 40)) {
        const li = el("li");
        const btn = el("button", "linkbtn state-dot-" + other.state, other.title);
        btn.type = "button";
        btn.addEventListener("click", () => {
          if (!active.includes(other)) { toggles.hubs.checked = true; rebuild(); }
          select(other);
          center(other);
        });
        li.append(btn);
        list.append(li);
      }
      if (items.length > 40) list.append(el("li", "more", "and " + (items.length - 40) + " more"));
      section.append(list);
      parts.push(section);
    }
    panel.replaceChildren(...parts);
  }

  function center(n) {
    view.x = canvas.clientWidth / 2 - n.x * view.k;
    view.y = canvas.clientHeight / 2 - n.y * view.k;
    dirty = true;
    draw();
  }

  function updateCounts() {
    const c = data.counts;
    const visible = active.length;
    const parts = [
      ["memos shown", visible + (visible < c.memos ? " of " + c.memos : "")],
      ["connections", activeEdges.length],
      ["superseded", c.superseded],
      ["pending", c.pending],
    ];
    if (matches) parts.push(["matching", matches.size]);
    countsEl.replaceChildren(...parts.flatMap(([label, value]) => [el("span", "count-label", label), el("strong", "", String(value))]));
  }

  search.addEventListener("input", () => {
    const q = search.value.trim().toLowerCase();
    matches = q ? new Set(active.filter((n) => (n.title + " " + n.id + " " + n.description).toLowerCase().includes(q)).map((n) => n.id)) : null;
    updateCounts();
    dirty = true;
    draw();
  });
  search.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && matches && matches.size) {
      const first = byId.get([...matches][0]);
      select(first);
      center(first);
    }
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "/" && document.activeElement !== search) { ev.preventDefault(); search.focus(); }
    if (ev.key === "Escape") { select(null); search.value = ""; matches = null; updateCounts(); draw(); }
  });
  for (const box of Object.values(toggles)) box.addEventListener("change", rebuild);
  window.addEventListener("resize", resize);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readPalette(); draw(); });

  fetch("/api/graph.json", { credentials: "same-origin" })
    .then((r) => { if (!r.ok) throw new Error("status " + r.status); return r.json(); })
    .then((payload) => {
      data = payload;
      nodes = payload.nodes;
      edges = payload.edges;
      byId = new Map(nodes.map((n) => [n.id, n]));
      readPalette();
      resize();
      renderPanel();
      rebuild();
      frame.classList.add("ready");
    })
    .catch((err) => {
      countsEl.replaceChildren(el("span", "error-inline", "The graph could not be loaded (" + err.message + ")."));
    });
})();
