const README_SOURCES = [
  { url: new URL("../README.md", window.location.href).href },
  {
    url: "https://api.github.com/repos/GiulioDER/RE-call/contents/README.md?ref=master",
    headers: { Accept: "application/vnd.github.raw+json" },
  },
  { url: "https://raw.githubusercontent.com/GiulioDER/RE-call/master/README.md" },
];
const MARKDOWN_RENDERER = "https://api.github.com/markdown/raw";
const TRANSLATION_ENDPOINT = document.querySelector("meta[name='translation-endpoint']")?.content;
const GOOGLE_TRANSLATION_ENDPOINT = "https://translate.googleapis.com/translate_a/single";
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
const TRANSLATION_SEPARATOR = "\n---9F3A---\n";
const ENGLISH = "english";
const BATCH_SIZE = 24;
const MAX_README_MARKDOWN_CHARS = 400_000;
const MAX_RENDERED_HTML_CHARS = 5_000_000;
const MAX_TRANSLATION_RESPONSE_CHARS = 2_000_000;
const TRANSLATION_TIMEOUT_MS = 3_000;
const TRANSLATION_ATTEMPTS = 2;
const TRANSLATION_TOTAL_TIMEOUT_MS = 15_000;
const STORAGE_KEY = "recall-readme-language";

const content = document.querySelector("#readme-content");
const language = document.querySelector("#readme-language");
const status = document.querySelector("#translation-status");
const state = {
  englishHtml: "",
  translated: new Map(),
  selectionVersion: 0,
  activeTranslation: null,
};

function setStatus(message, tone = "") {
  status.textContent = message;
  status.dataset.tone = tone;
}

function savedLanguage() {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch (error) {
    return null;
  }
}

function saveLanguage(locale) {
  try {
    window.localStorage.setItem(STORAGE_KEY, locale);
  } catch (error) {
    // Language selection remains functional when browser storage is unavailable.
  }
}

async function fetchText(url) {
  const source = typeof url === "object" ? url : { url };
  const response = await fetch(source.url, {
    headers: { Accept: "text/plain", ...source.headers },
  });
  if (!response.ok) {
    throw new Error("README source unavailable");
  }
  const contentLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(contentLength) && contentLength > MAX_README_MARKDOWN_CHARS) {
    throw new Error("README source is too large");
  }
  const text = await response.text();
  if (text.length > MAX_README_MARKDOWN_CHARS) {
    throw new Error("README source is too large");
  }
  return text;
}

async function loadMarkdown() {
  let markdownError;
  for (const source of README_SOURCES) {
    try {
      return await fetchText(source);
    } catch (error) {
      markdownError = error;
    }
  }
  throw markdownError || new Error("README source unavailable");
}

async function renderMarkdown(markdown) {
  const response = await fetch(MARKDOWN_RENDERER, {
    method: "POST",
    headers: {
      Accept: "text/html",
      "Content-Type": "text/plain; charset=utf-8",
    },
    body: markdown,
  });
  if (!response.ok) {
    throw new Error("README renderer unavailable");
  }
  const text = await response.text();
  if (text.length > MAX_RENDERED_HTML_CHARS) {
    throw new Error("Rendered README is too large");
  }
  return text;
}

function textNodes() {
  const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent || !node.nodeValue?.trim()) {
        return NodeFilter.FILTER_REJECT;
      }
      if (parent.closest("pre, code, script, style, .readme-no-translate")) {
        return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const nodes = [];
  let node;
  while ((node = walker.nextNode())) {
    nodes.push(node);
  }
  return nodes;
}

function cleanText(value) {
  return value.trim().replace(/\s+/g, " ");
}

async function translateBatch(texts, locale, signal) {
  if (!TRANSLATION_ENDPOINT) {
    throw new Error("translation endpoint is not configured");
  }
  let primaryError;
  for (let attempt = 0; attempt < TRANSLATION_ATTEMPTS; attempt += 1) {
    const body = new URLSearchParams({
      to: locale,
      text: JSON.stringify(texts),
    });
    const controller = new AbortController();
    const abort = () => controller.abort();
    if (signal?.aborted) {
      controller.abort();
    } else {
      signal?.addEventListener("abort", abort, { once: true });
    }
    const timeout = window.setTimeout(() => controller.abort(), TRANSLATION_TIMEOUT_MS);
    try {
      const response = await fetch(TRANSLATION_ENDPOINT, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body,
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error("translation provider unavailable");
      }
      const responseText = await response.text();
      if (responseText.length > MAX_TRANSLATION_RESPONSE_CHARS) {
        throw new Error("translation provider response is too large");
      }
      const payload = JSON.parse(responseText);
      if (
        payload?.result !== 1 ||
        !Array.isArray(payload.text) ||
        payload.text.length !== texts.length ||
        payload.text.some((item) => typeof item !== "string")
      ) {
        throw new Error("translation provider returned invalid text");
      }
      return payload.text;
    } catch (error) {
      primaryError = error;
      if (signal?.aborted || attempt === TRANSLATION_ATTEMPTS - 1) {
        break;
      }
    } finally {
      window.clearTimeout(timeout);
      signal?.removeEventListener("abort", abort);
    }
    await new Promise((resolve) => window.setTimeout(resolve, 300 * (attempt + 1)));
  }
  if (signal?.aborted) {
    throw primaryError || new Error("translation provider request failed");
  }
  return translateWithGoogle(texts, locale, signal);
}

async function translateWithGoogle(texts, locale, signal) {
  const target = GOOGLE_LOCALES[locale] || locale;
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (signal?.aborted) {
    controller.abort();
  } else {
    signal?.addEventListener("abort", abort, { once: true });
  }
  const timeout = window.setTimeout(() => controller.abort(), TRANSLATION_TIMEOUT_MS);
  try {
    const url = new URL(GOOGLE_TRANSLATION_ENDPOINT);
    url.search = new URLSearchParams({
      client: "gtx",
      sl: "auto",
      tl: target,
      dt: "t",
      q: texts.join(TRANSLATION_SEPARATOR),
    });
    const response = await fetch(url, {
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error("fallback translation provider unavailable");
    }
    const responseText = await response.text();
    if (responseText.length > MAX_TRANSLATION_RESPONSE_CHARS) {
      throw new Error("fallback translation response is too large");
    }
    const payload = JSON.parse(responseText);
    const combined = payload?.[0]
      ?.filter((segment) => Array.isArray(segment) && typeof segment[0] === "string")
      .map((segment) => segment[0])
      .join("");
    const translated = combined?.split(TRANSLATION_SEPARATOR);
    if (
      !Array.isArray(translated) ||
      translated.length !== texts.length ||
      translated.some((item) => typeof item !== "string")
    ) {
      throw new Error("fallback translation returned invalid text");
    }
    return translated;
  } finally {
    window.clearTimeout(timeout);
    signal?.removeEventListener("abort", abort);
  }
}

async function translateNodes(nodes, locale, signal) {
  if (nodes.length > 512) {
    throw new Error("README contains too many translatable text nodes");
  }
  const source = nodes.map((node) => cleanText(node.nodeValue || ""));
  const translated = [];
  let failedBatches = 0;
  const batchCount = Math.ceil(source.length / BATCH_SIZE);
  if (batchCount > 22) {
    throw new Error("README requires too many translation requests");
  }
  const batches = [];
  for (let start = 0; start < source.length; start += BATCH_SIZE) {
    batches.push(source.slice(start, start + BATCH_SIZE));
  }
  const results = await Promise.all(batches.map(async (batch) => {
    try {
      return { values: await translateBatch(batch, locale, signal), failed: false };
    } catch (error) {
      if (signal?.aborted) {
        throw error;
      }
      return { values: batch, failed: true };
    }
  }));
  for (const result of results) {
    translated.push(...result.values);
    if (result.failed) {
      failedBatches += 1;
    }
  }
  return { values: translated, failedBatches };
}

function applyTranslations(nodes, translated) {
  nodes.forEach((node, index) => {
    const raw = node.nodeValue || "";
    const leading = raw.match(/^\s*/)?.[0] || "";
    const trailing = raw.match(/\s*$/)?.[0] || "";
    node.nodeValue = `${leading}${translated[index]}${trailing}`;
  });
}

function restoreEnglish() {
  content.innerHTML = state.englishHtml;
  document.documentElement.lang = "en";
  setStatus("English source");
}

async function selectLanguage(locale) {
  const selectionVersion = ++state.selectionVersion;
  state.activeTranslation?.abort();
  if (locale === ENGLISH) {
    restoreEnglish();
    return;
  }

  restoreEnglish();
  setStatus("Translating the readable text…", "working");
  const nodes = textNodes();
  const controller = new AbortController();
  const translationDeadline = window.setTimeout(
    () => controller.abort(),
    TRANSLATION_TOTAL_TIMEOUT_MS,
  );
  state.activeTranslation = controller;
  try {
    let translated = state.translated.get(locale);
    let failedBatches = 0;
    if (!translated) {
      const result = await translateNodes(nodes, locale, controller.signal);
      translated = result.values;
      failedBatches = result.failedBatches;
      if (!failedBatches) {
        state.translated.set(locale, translated);
      }
    }
    if (selectionVersion !== state.selectionVersion || language.value !== locale) {
      return;
    }
    applyTranslations(nodes, translated);
    const selected = language.selectedOptions[0];
    document.documentElement.lang = selected.dataset.htmlLang || "en";
    if (failedBatches === Math.ceil(nodes.length / BATCH_SIZE)) {
      restoreEnglish();
      setStatus("Translation unavailable · showing English source", "error");
    } else if (failedBatches) {
      setStatus(
        `${selected.textContent} presentation · ${failedBatches} batch unavailable, English source kept`,
        "error",
      );
    } else {
      setStatus(`${selected.textContent} presentation · source remains English`);
    }
  } catch (error) {
    if (selectionVersion !== state.selectionVersion || language.value !== locale) {
      return;
    }
    restoreEnglish();
    setStatus("Translation unavailable · showing English source", "error");
  } finally {
    window.clearTimeout(translationDeadline);
    if (state.activeTranslation === controller) {
      state.activeTranslation = null;
    }
  }
}

async function start() {
  try {
    const markdown = await loadMarkdown();
    const html = await renderMarkdown(markdown);
    content.innerHTML = html;
    state.englishHtml = html;
    const saved = savedLanguage();
    const initial = [...language.options].some((option) => option.value === saved) ? saved : ENGLISH;
    language.value = initial;
    await selectLanguage(initial);
  } catch (error) {
    content.innerHTML = "<p class='readme-error'>The README could not be loaded. Use the source link above to open it directly on GitHub.</p>";
    setStatus("README unavailable", "error");
  }
}

language.addEventListener("change", () => {
  saveLanguage(language.value);
  void selectLanguage(language.value);
});

void start();
