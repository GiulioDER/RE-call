// RE-call memory graph: every memo a star, every declared link a line, in three dimensions.
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
    shares: document.getElementById("show-shares"),
    hubs: document.getElementById("show-hubs"),
    motion: document.getElementById("show-motion"),
  };
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const lensEl = document.getElementById("lens");
  const focusEl = document.getElementById("focus");
  const timeEl = document.getElementById("time");
  const timeLabel = document.getElementById("time-label");
  const playEl = document.getElementById("play");
  const legendEl = document.getElementById("legend");
  // A corpus read from the database (`/graph?corpus=<tenant>`); empty for the memo files. Its edge
  // kinds differ (`shares` instead of `pending`), so a toggle may be absent and reads as off.
  const corpus = canvas.dataset.corpus || "";
  const EDGE_KINDS = ["link", "supersedes", "pending", "shares"];
  const isOn = (kind) => Boolean(toggles[kind] && toggles[kind].checked);
  // Time travel: `asOf` is a YYYY-MM-DD day, or null for today.
  let asOf = null;
  let days = [];
  let playing = null;
  let supersededBy = new Map();
  let categories = { type: new Map(), folder: new Map() };
  let ageRank = new Map();

  let data = null;
  let nodes = [];
  let byId = new Map();
  let edges = [];
  let active = [];
  let activeEdges = [];
  let neighbours = new Map();
  let hover = null;
  let selected = null;
  let matches = null;
  let alpha = 0;
  let palette = {};
  let stars = [];

  // Camera: orbit around a pivot point, perspective projection.
  const cam = { yaw: 0.6, pitch: -0.32, dist: 900, zoom: 1, px: 0, py: 0, pz: 0 };
  let flight = null;
  let lastInteraction = 0;
  let projected = [];

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
      shares: v("--edge-shares"),
      select: v("--signal"),
      ink: v("--ink"),
      cats: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => v("--cat-" + i)),
      ageNew: v("--age-new"),
      ageOld: v("--age-old"),
    };
  }

  function rgb(hex) {
    const h = hex.replace("#", "");
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  }

  function mix(from, to, t) {
    const a = rgb(from), b = rgb(to);
    return "rgb(" + a.map((c, i) => Math.round(c + (b[i] - c) * t)).join(",") + ")";
  }

  // ------------------------------------------------------------ state over time, and lenses

  function stateAt(n) {
    if (!asOf) return n.state;
    if (n.valid_until && n.valid_until < asOf) return "expired";
    if (n.valid_from && n.valid_from > asOf) return "expired";
    const by = supersededBy.get(n.id);
    if (by && by.some((source) => byId.get(source).born <= asOf)) return "superseded";
    return "current";
  }

  function categoryOf(lens, n) {
    return lens === "type" ? n.type || "(no type)" : n.folder || "(top level)";
  }

  function colourOf(n, state) {
    const lens = lensEl.value;
    if (lens === "type" || lens === "folder") {
      const index = categories[lens].get(categoryOf(lens, n));
      return palette.cats[index === undefined ? 7 : Math.min(index, 7)];
    }
    if (lens === "age") return mix(palette.ageOld, palette.ageNew, ageRank.get(n.id) || 0);
    if (lens === "health") return n.issues.length ? palette.superseded : n.isolated ? palette.pending : palette.expired;
    return palette[state] || palette.current;
  }

  function legendItem(colour, label, ring) {
    const span = el("span");
    const dot = el("i", "dot");
    if (ring) dot.style.border = "1.5px solid " + colour;
    else dot.style.background = colour;
    span.append(dot, document.createTextNode(label));
    return span;
  }

  function renderLegend() {
    const lens = lensEl.value;
    let items;
    if (lens === "type" || lens === "folder") {
      items = [...categories[lens].entries()].slice(0, 8).map(([name, i]) => legendItem(palette.cats[Math.min(i, 7)], i >= 7 ? name + " and the rest" : name));
    } else if (lens === "age") {
      items = [legendItem(palette.ageNew, "newest"), legendItem(mix(palette.ageOld, palette.ageNew, 0.5), "middle"), legendItem(palette.ageOld, "oldest")];
    } else if (lens === "health") {
      items = [legendItem(palette.superseded, "has a lint issue"), legendItem(palette.pending, "linked only from index pages"), legendItem(palette.expired, "no issue found")];
    } else {
      items = [legendItem(palette.current, "current"), legendItem(palette.superseded, "superseded", true), legendItem(palette.expired, asOf ? "expired or not yet valid" : "expired"), legendItem(palette.pending, "pending review")];
    }
    legendEl.replaceChildren(...items);
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
    return Math.min(9, 2 + Math.sqrt(n.degree) * 0.9);
  }

  function moving() {
    return toggles.motion.checked && !reducedMotion.matches;
  }

  // ------------------------------------------------------------ layout, in three dimensions

  function rebuild() {
    const shown = new Set(EDGE_KINDS.filter(isOn));
    let pool = nodes.filter((n) => (toggles.hubs.checked || !n.hub) && (!asOf || n.born <= asOf));
    let live = new Set(pool.map((n) => n.id));
    // Claims waiting for review belong to today, so they are not drawn on a past date.
    let candidate = edges.filter((e) => shown.has(e.kind) && live.has(e.source) && live.has(e.target) && !(asOf && e.kind === "pending"));
    const hops = Number(focusEl.value);
    if (hops > 0 && selected && live.has(selected.id)) {
      const adjacent = new Map();
      for (const e of candidate) {
        if (!adjacent.has(e.source)) adjacent.set(e.source, []);
        if (!adjacent.has(e.target)) adjacent.set(e.target, []);
        adjacent.get(e.source).push(e.target);
        adjacent.get(e.target).push(e.source);
      }
      const keep = new Set([selected.id]);
      let frontier = [selected.id];
      for (let step = 0; step < hops; step++) {
        const next = [];
        for (const id of frontier) for (const other of adjacent.get(id) || []) if (!keep.has(other)) { keep.add(other); next.push(other); }
        frontier = next;
      }
      pool = pool.filter((n) => keep.has(n.id));
      live = keep;
      candidate = candidate.filter((e) => keep.has(e.source) && keep.has(e.target));
    }
    const wasEmpty = !active.length;
    active = pool;
    activeEdges = candidate;
    neighbours = new Map(active.map((n) => [n.id, new Set()]));
    for (const e of activeEdges) {
      neighbours.get(e.source).add(e.target);
      neighbours.get(e.target).add(e.source);
    }
    // A stable starting picture: the same memo starts in the same place on every load (a
    // Fibonacci sphere, radius growing with the corpus).
    const golden = Math.PI * (3 - Math.sqrt(5));
    const shell = 40 * Math.cbrt(active.length);
    let placed = 0;
    active.forEach((n, i) => {
      if (n.x === undefined) {
        placed++;
        const y = 1 - (2 * (i + 0.5)) / active.length;
        const r = Math.sqrt(1 - y * y);
        const a = i * golden;
        const s = shell * (0.55 + 0.45 * hash(n.id));
        n.x = Math.cos(a) * r * s;
        n.y = y * s;
        n.z = Math.sin(a) * r * s;
        n.vx = n.vy = n.vz = 0;
        n.phase = hash(n.id + "#") * Math.PI * 2;
      }
    });
    // Reheat only as much as the change needs, so a filter or a date step does not scramble the
    // picture the person is looking at.
    alpha = Math.max(alpha, wasEmpty ? 1 : placed ? 0.35 : 0.08);
    if (hops > 0 && selected) fitDistance();
    updateCounts();
    renderLegend();
  }

  function tick() {
    const cell = 110;
    const grid = new Map();
    const key = (x, y, z) => x + "," + y + "," + z;
    for (const n of active) {
      const k = key(Math.floor(n.x / cell), Math.floor(n.y / cell), Math.floor(n.z / cell));
      if (!grid.has(k)) grid.set(k, []);
      grid.get(k).push(n);
    }
    for (const n of active) {
      const gx = Math.floor(n.x / cell), gy = Math.floor(n.y / cell), gz = Math.floor(n.z / cell);
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        const bucket = grid.get(key(gx + dx, gy + dy, gz + dz));
        if (!bucket) continue;
        for (const m of bucket) {
          if (m === n) continue;
          const x = n.x - m.x, y = n.y - m.y, z = n.z - m.z;
          const d2 = x * x + y * y + z * z;
          if (d2 > cell * cell) continue;
          // Clamped at short range: two memos landing on one spot must not fling each other out.
          const f = (320 * alpha) / Math.max(d2, 49);
          n.vx += x * f; n.vy += y * f; n.vz += z * f;
        }
      }
    }
    for (const e of activeEdges) {
      const a = byId.get(e.source), b = byId.get(e.target);
      const x = b.x - a.x, y = b.y - a.y, z = b.z - a.z;
      const d = Math.sqrt(x * x + y * y + z * z) || 1;
      const rest = e.kind === "supersedes" ? 30 : e.kind === "shares" ? 90 : 70;
      const s = ((d - rest) / d) * (e.kind === "supersedes" ? 0.05 : 0.008) * alpha;
      a.vx += x * s; a.vy += y * s; a.vz += z * s;
      b.vx -= x * s; b.vy -= y * s; b.vz -= z * s;
    }
    for (const n of active) {
      n.vx -= n.x * 0.006 * alpha; n.vy -= n.y * 0.006 * alpha; n.vz -= n.z * 0.006 * alpha;
      n.vx *= 0.8; n.vy *= 0.8; n.vz *= 0.8;
      const speed = Math.hypot(n.vx, n.vy, n.vz);
      if (speed > 14) { n.vx *= 14 / speed; n.vy *= 14 / speed; n.vz *= 14 / speed; }
      n.x += n.vx; n.y += n.vy; n.z += n.vz;
    }
    alpha *= 0.985;
  }

  function fitDistance() {
    if (!active.length) return;
    // Orbit the middle of the memory, not the origin, and fit its body rather than its strays:
    // the 96th percentile distance from that centre fills a little under half the view.
    let sx = 0, sy = 0, sz = 0;
    for (const n of active) { sx += n.x; sy += n.y; sz += n.z; }
    cam.px = sx / active.length; cam.py = sy / active.length; cam.pz = sz / active.length;
    const ds = active.map((n) => Math.hypot(n.x - cam.px, n.y - cam.py, n.z - cam.pz)).sort((a, b) => a - b);
    const r = ds[Math.floor(ds.length * 0.96)] || 200;
    const view = Math.min(canvas.clientWidth, canvas.clientHeight) || 600;
    cam.dist = Math.max(160, r * 2.6);
    cam.zoom = (view * 0.44) / r;
  }

  // ------------------------------------------------------------ projection

  function project() {
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    projected = [];
    for (const n of active) {
      const x0 = n.x - cam.px, y0 = n.y - cam.py, z0 = n.z - cam.pz;
      const x1 = x0 * cy - z0 * sy;
      const z1 = x0 * sy + z0 * cy;
      const y2 = y0 * cp - z1 * sp;
      const z2 = y0 * sp + z1 * cp;
      const depth = cam.dist + z2;
      if (depth < 20) { n.visible = false; continue; }
      const s = (cam.dist / depth) * cam.zoom;
      n.sx = w / 2 + x1 * s;
      n.sy = h / 2 + y2 * s;
      n.scale = s;
      n.depth = depth;
      n.visible = true;
      projected.push(n);
    }
    projected.sort((a, b) => b.depth - a.depth);
  }

  function fog(n) {
    // Near memos are bright, far ones fade toward the ground colour.
    const near = cam.dist * 0.55, far = cam.dist * 1.6;
    return 1 - 0.78 * Math.min(1, Math.max(0, (n.depth - near) / (far - near)));
  }

  // ------------------------------------------------------------ drawing

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    stars = Array.from({ length: 140 }, (_, i) => ({
      x: hash("sx" + i) * canvas.clientWidth,
      y: hash("sy" + i) * canvas.clientHeight,
      r: 0.3 + hash("sr" + i) * 0.9,
      p: hash("sp" + i) * Math.PI * 2,
    }));
  }

  function focusSet() {
    if (selected) return new Set([selected.id, ...(neighbours.get(selected.id) || [])]);
    if (hover) return new Set([hover.id, ...(neighbours.get(hover.id) || [])]);
    return matches;
  }

  function draw(time) {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const t = time / 1000;
    ctx.clearRect(0, 0, w, h);

    // A faint field of fixed stars behind the memory, twinkling slowly.
    ctx.fillStyle = palette.ink;
    for (const s of stars) {
      ctx.globalAlpha = 0.05 + 0.06 * (moving() ? (1 + Math.sin(t * 0.7 + s.p)) / 2 : 0.5);
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
    }

    project();
    const focus = focusSet();

    for (const e of activeEdges) {
      const a = byId.get(e.source), b = byId.get(e.target);
      if (!a.visible || !b.visible) continue;
      const lit = !focus || (focus.has(a.id) && focus.has(b.id));
      const depthFade = (fog(a) + fog(b)) / 2;
      ctx.beginPath();
      ctx.setLineDash(e.kind === "pending" ? [5, 4] : e.kind === "shares" ? [2, 3] : []);
      if (e.kind === "link") {
        ctx.strokeStyle = palette.link;
        ctx.globalAlpha = (lit ? (focus ? 0.8 : 0.24) : 0.03) * depthFade;
        ctx.lineWidth = 0.7;
      } else if (e.kind === "shares") {
        ctx.strokeStyle = palette.shares;
        ctx.globalAlpha = (lit ? (focus ? 0.7 : 0.18) : 0.03) * depthFade;
        ctx.lineWidth = 0.6;
      } else if (e.kind === "supersedes") {
        ctx.strokeStyle = palette.supersedes;
        ctx.globalAlpha = (lit ? 0.85 : 0.1) * depthFade;
        ctx.lineWidth = 1.3;
      } else {
        ctx.strokeStyle = palette.pendingEdge;
        ctx.globalAlpha = (lit ? 0.95 : 0.2) * depthFade;
        ctx.lineWidth = 1.3;
        ctx.lineDashOffset = moving() ? -t * 18 : 0;
      }
      ctx.moveTo(a.sx, a.sy);
      ctx.lineTo(b.sx, b.sy);
      ctx.stroke();
      if (e.kind !== "link" && e.kind !== "shares" && lit) arrow(a, b);
      // A pulse of light runs from the newer memo to the one it replaces.
      if (e.kind === "supersedes" && lit && moving()) {
        const phase = (t * 0.45 + hash(e.source + e.target)) % 1;
        const x = a.sx + (b.sx - a.sx) * phase, y = a.sy + (b.sy - a.sy) * phase;
        const glow = ctx.createRadialGradient(x, y, 0, x, y, 5);
        glow.addColorStop(0, palette.supersedes);
        glow.addColorStop(1, "transparent");
        ctx.globalAlpha = 0.9 * depthFade;
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.setLineDash([]);
    ctx.lineDashOffset = 0;

    for (const n of projected) {
      const lit = !focus || focus.has(n.id);
      const r = Math.max(0.8, radius(n) * n.scale * 0.9);
      const f = fog(n);
      const state = stateAt(n);
      const lensOnState = lensEl.value === "state";
      if (lensOnState && state === "pending") {
        const breathe = moving() ? 2.3 + 0.7 * Math.sin(t * 2.2 + n.phase) : 2.6;
        const halo = ctx.createRadialGradient(n.sx, n.sy, 0, n.sx, n.sy, r * breathe * 1.6);
        halo.addColorStop(0, palette.pending);
        halo.addColorStop(1, "transparent");
        ctx.globalAlpha = (lit ? 0.35 : 0.06) * f;
        ctx.fillStyle = halo;
        ctx.beginPath();
        ctx.arc(n.sx, n.sy, r * breathe * 1.6, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = (lit ? 1 : 0.12) * f;
      ctx.beginPath();
      ctx.arc(n.sx, n.sy, r, 0, Math.PI * 2);
      const colour = colourOf(n, state);
      if (lensOnState && state === "superseded") {
        ctx.strokeStyle = colour;
        ctx.lineWidth = 1.3;
        ctx.stroke();
      } else {
        ctx.fillStyle = colour;
        ctx.fill();
        if (lit && lensOnState && state === "current" && r > 2.5) {
          // A soft bloom on the larger, nearer memos gives the sphere its depth.
          ctx.globalAlpha = 0.18 * f;
          const bloom = ctx.createRadialGradient(n.sx, n.sy, r * 0.4, n.sx, n.sy, r * 2.6);
          bloom.addColorStop(0, palette.current);
          bloom.addColorStop(1, "transparent");
          ctx.fillStyle = bloom;
          ctx.beginPath();
          ctx.arc(n.sx, n.sy, r * 2.6, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      if (n === selected || n === hover) {
        ctx.beginPath();
        ctx.strokeStyle = palette.select;
        ctx.globalAlpha = 1;
        ctx.lineWidth = 1.6;
        ctx.arc(n.sx, n.sy, r + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  function arrow(a, b) {
    const angle = Math.atan2(b.sy - a.sy, b.sx - a.sx);
    const r = radius(b) * b.scale * 0.9 + 2;
    const tipX = b.sx - Math.cos(angle) * r, tipY = b.sy - Math.sin(angle) * r;
    const size = 6;
    ctx.beginPath();
    ctx.setLineDash([]);
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX - size * Math.cos(angle - 0.45), tipY - size * Math.sin(angle - 0.45));
    ctx.lineTo(tipX - size * Math.cos(angle + 0.45), tipY - size * Math.sin(angle + 0.45));
    ctx.closePath();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fill();
  }

  // ------------------------------------------------------------ animation

  let needsDraw = true;
  function frameLoop(time) {
    if (alpha > 0.004) {
      for (let i = 0; i < 3; i++) tick();
      if (!selected && !flight) fitDistance();
      needsDraw = true;
    }
    if (flight) {
      const k = Math.min(1, (time - flight.start) / flight.ms);
      const ease = 1 - Math.pow(1 - k, 3);
      for (const axis of ["px", "py", "pz", "dist"]) cam[axis] = flight.from[axis] + (flight.to[axis] - flight.from[axis]) * ease;
      if (k >= 1) flight = null;
      needsDraw = true;
    }
    if (moving()) {
      // Drift slowly while nobody is steering; stop the moment someone is.
      if (!drag && time - lastInteraction > 2500) cam.yaw += 0.0011;
      needsDraw = true;
    }
    if (needsDraw) {
      needsDraw = false;
      draw(time);
    }
    requestAnimationFrame(frameLoop);
  }

  function flyTo(n) {
    flight = {
      start: performance.now(),
      ms: 700,
      from: { px: cam.px, py: cam.py, pz: cam.pz, dist: cam.dist },
      to: { px: n.x, py: n.y, pz: n.z, dist: Math.max(220, cam.dist * 0.55) },
    };
  }

  // ------------------------------------------------------------ interaction

  function at(px, py) {
    let best = null, bestD = Infinity;
    for (let i = projected.length - 1; i >= 0; i--) {
      const n = projected[i];
      const d = (n.sx - px) ** 2 + (n.sy - py) ** 2;
      const reach = (Math.max(4, radius(n) * n.scale) + 5) ** 2;
      if (d < reach && d < bestD) { best = n; bestD = d; }
    }
    return best;
  }

  let drag = null;
  canvas.addEventListener("pointerdown", (ev) => {
    drag = { x: ev.clientX, y: ev.clientY, yaw: cam.yaw, pitch: cam.pitch, moved: false };
    lastInteraction = performance.now();
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener("pointermove", (ev) => {
    const box = canvas.getBoundingClientRect();
    if (drag) {
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      cam.yaw = drag.yaw + dx * 0.006;
      cam.pitch = Math.max(-1.4, Math.min(1.4, drag.pitch + dy * 0.006));
      lastInteraction = performance.now();
      needsDraw = true;
      return;
    }
    const found = at(ev.clientX - box.left, ev.clientY - box.top);
    if (found !== hover) {
      hover = found;
      canvas.style.cursor = found ? "pointer" : "grab";
      needsDraw = true;
    }
    showTip(found, ev.clientX - box.left, ev.clientY - box.top);
  });
  canvas.addEventListener("pointerup", (ev) => {
    const box = canvas.getBoundingClientRect();
    if (drag && !drag.moved) select(at(ev.clientX - box.left, ev.clientY - box.top), true);
    drag = null;
  });
  canvas.addEventListener("pointerleave", () => {
    hover = null;
    tip.hidden = true;
    needsDraw = true;
  });
  canvas.addEventListener("dblclick", () => {
    select(null);
    flight = { start: performance.now(), ms: 700, from: { px: cam.px, py: cam.py, pz: cam.pz, dist: cam.dist }, to: { px: 0, py: 0, pz: 0, dist: cam.dist } };
    setTimeout(fitDistance, 720);
  });
  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    cam.zoom = Math.max(0.1, Math.min(14, cam.zoom * Math.exp(-ev.deltaY * 0.0015)));
    lastInteraction = performance.now();
    needsDraw = true;
  }, { passive: false });

  const STATE_LABEL = { current: "current", superseded: "superseded", expired: "expired", pending: "pending review" };

  function showTip(n, x, y) {
    if (!n) { tip.hidden = true; return; }
    tip.replaceChildren(el("strong", "", n.title), el("span", "tip-meta", STATE_LABEL[stateAt(n)] + " · " + n.degree + " connections" + (n.issues.length ? " · " + n.issues.length + " issue(s)" : "")));
    tip.hidden = false;
    const w = frame.clientWidth;
    tip.style.left = Math.min(x + 14, w - tip.offsetWidth - 8) + "px";
    tip.style.top = y + 14 + "px";
  }

  function select(n, fly) {
    selected = n;
    if (Number(focusEl.value) > 0) rebuild();
    if (n && fly) flyTo(n);
    needsDraw = true;
    renderPanel();
  }

  function renderPanel() {
    if (!selected) {
      panel.replaceChildren(el("p", "panel-empty", "Select a star to read the memo and follow its links. Drag to orbit, scroll to zoom, double-click to reset."));
      return;
    }
    const n = selected;
    const head = el("div", "panel-head");
    head.append(el("span", "state state-" + stateAt(n), STATE_LABEL[stateAt(n)]));
    if (n.type) head.append(el("span", "chip", n.type));
    const close = el("button", "close", "×");
    close.type = "button";
    close.setAttribute("aria-label", "Close");
    close.addEventListener("click", () => select(null));
    head.append(close);
    const parts = [head, el("h2", "", n.title), el("code", "path", n.id)];
    const validity = [n.valid_from ? "valid from " + n.valid_from : "", n.valid_until ? "until " + n.valid_until : ""].filter(Boolean).join(" ");
    parts.push(el("p", "meta-line", "dated " + n.born + (validity ? " · " + validity : "") + (n.isolated ? " · linked only from index pages" : "")));
    if (n.description) parts.push(el("p", "desc", n.description));
    if (n.issues.length) {
      const list = el("ul", "issues");
      for (const issue of n.issues) {
        const li = el("li");
        li.append(el("code", "", issue.code), document.createTextNode(issue.message));
        list.append(li);
      }
      parts.push(list);
    }
    const actions = el("div", "panel-actions");
    if (corpus) {
      // A database file has no memo on this machine: its page is View 2.
      const open = el("a", "review-link", "Open this file's page →");
      open.href = "/corpus/source?tenant=" + encodeURIComponent(corpus) + "&file=" + encodeURIComponent(n.id);
      actions.append(open);
    } else {
      const read = el("a", "review-link", "Read this memo →");
      read.href = "/read?path=" + encodeURIComponent(n.id);
      const edit = el("a", "review-link", "Edit supersession, validity, status →");
      edit.href = "/memo?path=" + encodeURIComponent(n.id);
      actions.append(read, edit);
    }
    if (n.claim) {
      const review = el("a", "review-link", "Review the pending claim →");
      review.href = "/review?claim=" + encodeURIComponent(n.claim);
      actions.append(review);
    }
    parts.push(actions);
    const groups = [
      ["Supersedes", (e) => e.kind === "supersedes" && e.source === n.id, "target"],
      ["Superseded by", (e) => e.kind === "supersedes" && e.target === n.id, "source"],
      ["Would be replaced by (pending)", (e) => e.kind === "pending" && e.target === n.id, "source"],
      ["Would replace (pending)", (e) => e.kind === "pending" && e.source === n.id, "target"],
      ["Links to", (e) => e.kind === "link" && e.source === n.id, "target"],
      ["Linked from", (e) => e.kind === "link" && e.target === n.id, "source"],
      ["Shares names with", (e) => e.kind === "shares" && e.source === n.id, "target"],
      ["Shares names with", (e) => e.kind === "shares" && e.target === n.id, "source"],
    ];
    for (const [label, test, end] of groups) {
      const items = edges.filter(test).map((e) => byId.get(e[end])).filter(Boolean);
      if (!items.length) continue;
      const section = el("section", "links");
      section.append(el("h3", "", label + " · " + items.length));
      const list = el("ul");
      for (const other of items.slice(0, 40)) {
        const li = el("li");
        const btn = el("button", "linkbtn state-dot-" + stateAt(other), other.title);
        btn.type = "button";
        btn.addEventListener("click", () => {
          if (!active.includes(other)) { toggles.hubs.checked = true; rebuild(); }
          select(other, true);
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

  function updateCounts() {
    const c = data.counts;
    const visible = active.length;
    const parts = [
      [(corpus ? "files" : "memos") + " shown", visible + (visible < c.memos ? " of " + c.memos : "")],
      ["connections", activeEdges.length],
      // Counted over what is on screen, at the date shown, never the totals for today.
      ["superseded", active.filter((n) => stateAt(n) === "superseded").length],
    ];
    if (!asOf) parts.push(["pending", active.filter((n) => n.state === "pending").length]);
    if (lensEl.value === "health") {
      parts.push(
        ["with issues", active.filter((n) => n.issues.length).length],
        ["linked only from index pages", active.filter((n) => n.isolated).length],
      );
    }
    if (asOf) parts.unshift(["as of", asOf]);
    if (matches) parts.push(["matching", matches.size]);
    countsEl.replaceChildren(...parts.flatMap(([label, value]) => [el("span", "count-label", label), el("strong", "", String(value))]));
  }

  search.addEventListener("input", () => {
    const q = search.value.trim().toLowerCase();
    matches = q ? new Set(active.filter((n) => (n.title + " " + n.id + " " + n.description).toLowerCase().includes(q)).map((n) => n.id)) : null;
    updateCounts();
    needsDraw = true;
  });
  search.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && matches && matches.size) select(byId.get([...matches][0]), true);
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "/" && document.activeElement !== search) { ev.preventDefault(); search.focus(); }
    if (ev.key === "Escape") { select(null); search.value = ""; matches = null; updateCounts(); needsDraw = true; }
  });
  for (const name of [...EDGE_KINDS, "hubs"]) if (toggles[name]) toggles[name].addEventListener("change", rebuild);
  toggles.motion.addEventListener("change", () => { needsDraw = true; });
  lensEl.addEventListener("change", () => { renderLegend(); updateCounts(); needsDraw = true; });
  focusEl.addEventListener("change", () => { rebuild(); if (!selected) fitDistance(); needsDraw = true; });

  function setTime(index) {
    timeEl.value = String(index);
    asOf = index >= days.length - 1 ? null : days[index];
    timeLabel.textContent = asOf || "today";
    rebuild();
    needsDraw = true;
  }
  timeEl.addEventListener("input", () => { stopPlaying(); setTime(Number(timeEl.value)); });
  function stopPlaying() {
    if (playing) { clearInterval(playing); playing = null; playEl.textContent = "▶"; }
  }
  playEl.addEventListener("click", () => {
    if (playing) { stopPlaying(); return; }
    let index = Number(timeEl.value) >= days.length - 1 ? 0 : Number(timeEl.value);
    playEl.textContent = "❚❚";
    setTime(index);
    // A day per tick: the memory grows, and supersessions arrive as the newer memos do.
    playing = setInterval(() => {
      index += 1;
      setTime(index);
      if (index >= days.length - 1) stopPlaying();
    }, reducedMotion.matches ? 600 : 160);
  });
  window.addEventListener("resize", () => { resize(); needsDraw = true; });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readPalette(); needsDraw = true; });

  fetch("/api/graph.json" + (corpus ? "?corpus=" + encodeURIComponent(corpus) : ""), { credentials: "same-origin" })
    .then((r) => { if (!r.ok) throw new Error("status " + r.status); return r.json(); })
    .then((payload) => {
      data = payload;
      nodes = payload.nodes;
      edges = payload.edges;
      byId = new Map(nodes.map((n) => [n.id, n]));
      supersededBy = new Map();
      for (const e of edges) if (e.kind === "supersedes") {
        if (!supersededBy.has(e.target)) supersededBy.set(e.target, []);
        supersededBy.get(e.target).push(e.source);
      }
      for (const lens of ["type", "folder"]) {
        const tally = new Map();
        for (const n of nodes) tally.set(categoryOf(lens, n), (tally.get(categoryOf(lens, n)) || 0) + 1);
        categories[lens] = new Map([...tally.entries()].sort((a, b) => b[1] - a[1]).map(([name], i) => [name, i]));
      }
      const born = [...new Set(nodes.map((n) => n.born))].sort();
      ageRank = new Map(nodes.map((n) => [n.id, born.length > 1 ? born.indexOf(n.born) / (born.length - 1) : 1]));
      days = [...born, "today"];
      timeEl.max = String(days.length - 1);
      timeEl.value = String(days.length - 1);
      readPalette();
      resize();
      renderPanel();
      rebuild();
      fitDistance();
      frame.classList.add("ready");
      requestAnimationFrame(frameLoop);
      // `/graph#memo=<path>` opens on that memo, as the editing page links back here.
      const wanted = location.hash.startsWith("#memo=") ? decodeURIComponent(location.hash.slice(6)) : "";
      if (wanted && byId.has(wanted)) {
        const target = byId.get(wanted);
        if (!active.includes(target)) { toggles.hubs.checked = true; rebuild(); }
        setTimeout(() => select(target, true), 900);
      }
    })
    .catch((err) => {
      countsEl.replaceChildren(el("span", "error-inline", "The graph could not be loaded (" + err.message + ")."));
    });
})();
