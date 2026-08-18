/* Shared presentation translation for the setup pages.

   English remains the source of truth. The control translates readable page
   copy only, leaving commands, code, and the README's dedicated renderer
   untouched. A small endpoint is tried first, with Google's public fallback
   keeping the static site useful when the primary service is unavailable.
*/
(() => {
  "use strict";

  if (document.body.classList.contains("readme-page")) return;

  const control = document.querySelector("[data-translate-control]");
  if (!control) return;

  const language = control.querySelector("select");
  const status = control.querySelector("[role='status']");
  if (!language || !status) return;

  const TRANSLATION_ENDPOINT = "https://api.translate.zvo.cn/translate.json";
  const GOOGLE_ENDPOINT = "https://translate.googleapis.com/translate_a/single";
  const GOOGLE_LOCALES = {
    italian: "it",
    spanish: "es",
    french: "fr",
    german: "de",
    portuguese: "pt",
    chinese_simplified: "zh-CN",
    japanese: "ja",
    korean: "ko",
    russian: "ru",
    arabic: "ar",
    hindi: "hi",
    turkish: "tr",
  };
  const ENGLISH = "english";
  const BATCH_SIZE = 24;
  const MAX_NODES = 512;
  const MAX_BATCHES = 22;
  const REQUEST_TIMEOUT_MS = 3_000;
  const TOTAL_TIMEOUT_MS = 15_000;
  const SEPARATOR = "\n---9F3A---\n";
  const STORAGE_KEY = "recall-language";

  const original = new Map();
  const translated = new Map();
  let activeController = null;
  let selectionVersion = 0;

  function readableTextNodes() {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent || !node.nodeValue?.trim()) return NodeFilter.FILTER_REJECT;
        if (
          parent.closest(
            "pre, code, script, style, select, option, [data-no-translate], " +
              "[data-translate-control], .terminal, .copy-button, [data-rail-progress]",
          )
        ) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const nodes = [];
    let node;
    while ((node = walker.nextNode())) nodes.push(node);
    return nodes;
  }

  function clean(value) {
    return value.trim().replace(/\s+/g, " ");
  }

  function setStatus(message, tone = "") {
    status.textContent = message;
    status.dataset.tone = tone;
  }

  function savedLanguage() {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch {
      return null;
    }
  }

  function saveLanguage(locale) {
    try {
      window.localStorage.setItem(STORAGE_KEY, locale);
    } catch {
      /* Translation remains functional when storage is unavailable. */
    }
  }

  function requestWithTimeout(url, options, signal) {
    const controller = new AbortController();
    const abort = () => controller.abort();
    if (signal?.aborted) controller.abort();
    else signal?.addEventListener("abort", abort, { once: true });
    const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    return fetch(url, { ...options, signal: controller.signal }).finally(() => {
      window.clearTimeout(timeout);
      signal?.removeEventListener("abort", abort);
    });
  }

  async function translateWithGoogle(texts, locale, signal) {
    const url = new URL(GOOGLE_ENDPOINT);
    url.search = new URLSearchParams({
      client: "gtx",
      sl: "auto",
      tl: GOOGLE_LOCALES[locale] || locale,
      dt: "t",
      q: texts.join(SEPARATOR),
    });
    const response = await requestWithTimeout(url, { headers: { Accept: "application/json" } }, signal);
    if (!response.ok) throw new Error("fallback translation unavailable");
    const payload = await response.json();
    const combined = payload?.[0]
      ?.filter((segment) => Array.isArray(segment) && typeof segment[0] === "string")
      .map((segment) => segment[0])
      .join("");
    const result = combined?.split(SEPARATOR);
    if (!Array.isArray(result) || result.length !== texts.length) {
      throw new Error("fallback translation returned invalid text");
    }
    return result;
  }

  async function translateBatch(texts, locale, signal) {
    try {
      const body = new URLSearchParams({ to: locale, text: JSON.stringify(texts) });
      const response = await requestWithTimeout(
        TRANSLATION_ENDPOINT,
        {
          method: "POST",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
          },
          body,
        },
        signal,
      );
      if (!response.ok) throw new Error("translation provider unavailable");
      const payload = await response.json();
      if (
        payload?.result === 1 &&
        Array.isArray(payload.text) &&
        payload.text.length === texts.length &&
        payload.text.every((item) => typeof item === "string")
      ) {
        return payload.text;
      }
      throw new Error("translation provider returned invalid text");
    } catch (error) {
      if (signal?.aborted) throw error;
      return translateWithGoogle(texts, locale, signal);
    }
  }

  async function translateNodes(nodes, locale, signal) {
    if (nodes.length > MAX_NODES) throw new Error("page contains too much readable text");
    const source = nodes.map((node) => clean(original.get(node) || node.nodeValue || ""));
    const batchCount = Math.ceil(source.length / BATCH_SIZE);
    if (batchCount > MAX_BATCHES) throw new Error("page requires too many translation requests");

    const batches = [];
    for (let start = 0; start < source.length; start += BATCH_SIZE) {
      batches.push(source.slice(start, start + BATCH_SIZE));
    }
    const results = await Promise.all(
      batches.map((batch) => translateBatch(batch, locale, signal)),
    );
    return results.flat();
  }

  function restoreEnglish(nodes) {
    nodes.forEach((node) => {
      if (original.has(node)) node.nodeValue = original.get(node);
    });
    document.documentElement.lang = "en";
    setStatus("English source");
  }

  function apply(nodes, values, locale) {
    nodes.forEach((node, index) => {
      const raw = original.get(node) || "";
      const leading = raw.match(/^\s*/)?.[0] || "";
      const trailing = raw.match(/\s*$/)?.[0] || "";
      node.nodeValue = `${leading}${values[index]}${trailing}`;
    });
    const selected = language.selectedOptions[0];
    document.documentElement.lang = selected.dataset.htmlLang || locale;
    setStatus(`${selected.textContent} presentation · source remains English`);
  }

  async function selectLanguage(locale, nodes) {
    const version = ++selectionVersion;
    activeController?.abort();
    restoreEnglish(nodes);
    if (locale === ENGLISH) return;

    setStatus("Translating the readable text…", "working");
    const controller = new AbortController();
    const deadline = window.setTimeout(() => controller.abort(), TOTAL_TIMEOUT_MS);
    activeController = controller;
    try {
      let values = translated.get(locale);
      if (!values) {
        values = await translateNodes(nodes, locale, controller.signal);
        translated.set(locale, values);
      }
      if (version !== selectionVersion || language.value !== locale) return;
      apply(nodes, values, locale);
    } catch (error) {
      if (version !== selectionVersion || language.value !== locale) return;
      restoreEnglish(nodes);
      setStatus("Translation unavailable · showing English source", "error");
    } finally {
      window.clearTimeout(deadline);
      if (activeController === controller) activeController = null;
    }
  }

  function init() {
    const nodes = readableTextNodes();
    nodes.forEach((node) => original.set(node, node.nodeValue));
    const saved = savedLanguage();
    const initial = [...language.options].some((option) => option.value === saved)
      ? saved
      : ENGLISH;
    language.value = initial;
    language.addEventListener("change", () => {
      saveLanguage(language.value);
      void selectLanguage(language.value, nodes);
    });
    void selectLanguage(initial, nodes);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
