"""Exercise README loading, URL resolution and restoration through the shipped JS.

BUG-002/003 red proof: baseline 61b428432 fails the relative-resource assertion
and all four abort assertions. Header and body cases use native Fetch against a
loopback HTTP server. Unchanged nodes pass after repair; receipts retain outcomes.
"""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

NODE = shutil.which("node")
SCRIPT = Path(__file__).resolve().parents[1] / "site" / "readme.js"

HARNESS = r"""
const assert = require("node:assert/strict");
const http = require("node:http");
const vm = require("node:vm");
const { setTimeout: delay, setImmediate: turn } = require("node:timers/promises");
const input = JSON.parse(require("node:fs").readFileSync(0, "utf8"));

async function main() {
  const timers = new Map();
  const requests = [];
  let nextTimer = 0;
  let onChange;
  let elements = [];
  let html = "Loading";
  const content = {
    get innerHTML() {
      return elements.length ? elements.map((node) =>
        `<${node.tag} ${node.attr}="${node.value}"></${node.tag}>`).join("") : html;
    },
    set innerHTML(value) {
      html = value;
      elements = [...value.matchAll(/<(a|img)\b[^>]*\b(href|src)="([^"]*)"/g)].map((match) => ({
        tag: match[1], attr: match[2], value: match[3],
        getAttribute(name) { return name === this.attr ? this.value : null; },
        setAttribute(name, value) { assert.equal(name, this.attr); this.value = value; },
      }));
    },
    querySelectorAll(selector) { return elements.filter((node) => selector.startsWith(node.tag + "[")); },
  };
  const status = { textContent: "", dataset: {} };
  const language = {
    value: "english", options: [{ value: "english" }],
    addEventListener(_name, callback) { onChange = callback; },
  };
  const document = {
    documentElement: { lang: "en" },
    querySelector(selector) {
      return { "#readme-content": content, "#readme-language": language,
        "#translation-status": status }[selector] || null;
    },
    createTreeWalker() { return { nextNode: () => null }; },
  };
  const rendered = '<a href="docs/API.md">API</a><img src="docs/postgresql-badge.svg">' +
    '<a href="#install">Install</a><a href="https://example.test/docs">External</a>' +
    '<img src="//example.test/badge.svg"><a href="mailto:reader@example.test">Mail</a>';
  let sourceCount = 0;
  const server = http.createServer((req, res) => {
    req.resume();
    const request = requests.find((item) => item.path === req.url && !item.received);
    request.received = true;
    const stalled = input.scenario.startsWith("renderer") ? req.url === "/renderer" :
      input.scenario.startsWith("source") && req.url === "/source1";
    if (stalled) {
      if (input.scenario.endsWith("body")) {
        res.writeHead(200); res.flushHeaders();
      }
      return; // Deliberately never finish the headers or body.
    }
    res.writeHead(200);
    res.end(req.url === "/renderer" ? rendered : "# README");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = "http://127.0.0.1:" + server.address().port;
  async function until(predicate) {
    for (let i = 0; i < 200 && !predicate(); ++i) await delay(10);
    return predicate();
  }
  try {
    vm.runInNewContext(input.source, {
      document, AbortController, URL, URLSearchParams,
      NodeFilter: { SHOW_TEXT: 4, FILTER_ACCEPT: 1, FILTER_REJECT: 2 },
      window: {
        location: { href: "https://giulioder.github.io/RE-call/readme.html" },
        localStorage: { getItem: () => null, setItem: () => {} },
        setTimeout(callback, ms) { timers.set(++nextTimer, { callback, ms }); return nextTimer; },
        clearTimeout(id) { timers.delete(id); },
      },
      async fetch(url, options) {
        const path = String(url).includes("/markdown/raw") ? "/renderer" : "/source" + (++sourceCount);
        const request = { path, signal: options.signal, headers: false, received: false };
        requests.push(request);
        const response = await fetch(base + path, options);
        request.headers = true;
        return response;
      },
    });
    if (input.scenario === "links") {
      assert(await until(() => status.textContent === "English source"), "fixture must finish startup");
      const expected = [
        "https://github.com/GiulioDER/RE-call/blob/master/docs/API.md",
        "https://raw.githubusercontent.com/GiulioDER/RE-call/master/docs/postgresql-badge.svg",
        "#install", "https://example.test/docs", "//example.test/badge.svg", "mailto:reader@example.test",
      ];
      assert.deepEqual(elements.map((node) => node.value), expected,
        "README resources must resolve against the repository");
      onChange();
      await turn();
      assert.deepEqual(elements.map((node) => node.value), expected,
        "restoring English must retain repository resource URLs");
    } else {
      const path = input.scenario.startsWith("renderer") ? "/renderer" : "/source1";
      assert(await until(() => requests.some((item) => item.path === path && item.received)),
        "fixture must reach the stalled HTTP request");
      const request = requests.find((item) => item.path === path);
      if (input.scenario.endsWith("body")) {
        assert(await until(() => request.headers), "fixture must reach body reading");
      }
      await turn();
      for (const [id, timer] of [...timers]) { timers.delete(id); timer.callback(); }
      await turn();
      assert(request.signal?.aborted, "stalled README headers and body must receive abort");
      if (path === "/source1") {
        assert(await until(() => status.textContent === "English source"),
          "source timeout must advance to a working fallback");
        assert.equal(sourceCount, 2);
      } else {
        assert(await until(() => status.textContent === "README unavailable"),
          "renderer timeout must show the error state");
        assert(content.innerHTML.includes("README could not be loaded"));
      }
    }
    assert.equal(timers.size, 0, "completed README requests must release timers");
  } finally {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.skipif(NODE is None, reason="README runtime checks require Node.js")
@pytest.mark.parametrize("scenario", ["links", "source-headers", "source-body", "renderer-headers", "renderer-body"])
def test_readme_resources_and_loading(scenario):
    result = subprocess.run(
        [NODE, "-e", HARNESS],
        input=json.dumps({"source": SCRIPT.read_text(encoding="utf-8"), "scenario": scenario}),
        text=True, capture_output=True, timeout=15, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
