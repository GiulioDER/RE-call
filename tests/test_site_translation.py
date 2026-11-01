"""Exercise the shipped translation control with real, stalled HTTP response bodies.

BUG-001: request and selection deadlines must protect JSON consumption, not just
response headers. Red-proof baseline: source 01ab25f374 at requestWithTimeout. Each parameter
node must fail the intended stalled-body abort assertion before the repair,
then pass unchanged with the fix. The audit receipt records observed outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

NODE = shutil.which("node")
SCRIPT = Path(__file__).resolve().parents[1] / "site" / "translate.js"

HARNESS = r"""
const assert = require("node:assert/strict");
const http = require("node:http");
const vm = require("node:vm");
const { setTimeout: delay, setImmediate: turn } = require("node:timers/promises");
const input = JSON.parse(require("node:fs").readFileSync(0, "utf8"));

async function main() {
  const requests = [];
  const timers = new Map();
  let nextTimer = 0;
  let onChange;
  const status = { textContent: "", dataset: {} };
  const text = { nodeValue: " Hello ", parentElement: { closest: () => false } };
  const language = {
    value: "english",
    options: [{ value: "english" }, { value: "italian" }],
    selectedOptions: [{ textContent: "Italian", dataset: { htmlLang: "it" } }],
    addEventListener: (_name, callback) => { onChange = callback; },
  };
  const document = {
    readyState: "complete",
    body: { classList: { contains: () => false } },
    documentElement: { lang: "en" },
    querySelector: () => ({
      querySelector: (selector) => selector === "select" ? language : status,
    }),
    createTreeWalker: (_body, _kind, filter) => {
      let visited = false;
      return { nextNode: () => {
        if (visited) return null;
        visited = true;
        return filter.acceptNode(text) === 1 ? text : null;
      } };
    },
  };
  const server = http.createServer((req, res) => {
    req.resume();
    if (input.scenario === "fallback" && req.url === "/primary") {
      res.writeHead(503).end();
    } else if (input.scenario === "primary" && req.url === "/fallback") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify([[["Ciao"]]]));
    } else {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.flushHeaders(); // A valid response whose body deliberately never ends.
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = "http://127.0.0.1:" + server.address().port;
  async function until(predicate) {
    for (let i = 0; i < 200 && !predicate(); ++i) await delay(10);
    return predicate();
  }
  function fire(milliseconds) {
    for (const [id, timer] of [...timers]) {
      if (timer.ms === milliseconds) {
        timers.delete(id);
        timer.callback();
      }
    }
  }
  try {
    vm.runInNewContext(input.source, {
      document, AbortController, URL, URLSearchParams,
      NodeFilter: { SHOW_TEXT: 4, FILTER_ACCEPT: 1, FILTER_REJECT: 2 },
      window: {
        localStorage: { getItem: () => null, setItem: () => {} },
        setTimeout: (callback, ms) => {
          timers.set(++nextTimer, { callback, ms });
          return nextTimer;
        },
        clearTimeout: (id) => timers.delete(id),
      },
      fetch: async (url, options) => {
        const path = String(url).includes("googleapis") ? "/fallback" : "/primary";
        const request = { signal: options.signal, headers: false, path };
        requests.push(request);
        const response = await fetch(base + path, options);
        request.headers = true;
        return response;
      },
    });
    language.value = "italian";
    onChange();
    const index = input.scenario === "fallback" ? 1 : 0;
    assert(await until(() => requests[index]?.headers), "fixture must reach body reading");
    await turn();
    await turn();
    if (input.scenario === "selection") {
      language.value = "english";
      onChange();
    } else {
      fire(input.scenario === "total" ? 15000 : 3000);
    }
    await turn();
    assert(requests[index].signal.aborted, "stalled response body must receive abort");
    if (input.scenario === "primary") {
      await until(() => status.textContent.includes("presentation"));
      assert.equal(text.nodeValue, " Ciao ", "request deadline must allow fallback");
      assert.equal(document.documentElement.lang, "it");
    } else if (input.scenario === "selection") {
      assert.equal(text.nodeValue, " Hello ");
      assert.equal(status.textContent, "English source");
    } else {
      await until(() => status.dataset.tone === "error");
      assert.equal(text.nodeValue, " Hello ");
      assert.equal(status.textContent, "Translation unavailable · showing English source");
      assert.equal(requests.length, input.scenario === "fallback" ? 2 : 1);
    }
    await until(() => timers.size === 0);
    assert.equal(timers.size, 0, "settled translation must release its timers");
  } finally {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
"""


@pytest.mark.skipif(NODE is None, reason="site translation runtime checks require Node.js")
@pytest.mark.parametrize("scenario", ["primary", "fallback", "total", "selection"])
def test_translation_deadlines_cover_response_bodies(scenario: str) -> None:
    result = subprocess.run(
        [NODE, "-e", HARNESS],
        input=json.dumps({"source": SCRIPT.read_text(encoding="utf-8"), "scenario": scenario}),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

