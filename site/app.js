/* RE-call setup guide, behaviour layer.

   Four jobs, no dependencies:
     1. OS tabs switch every block on the page at once, and the choice sticks.
     2. Copy buttons copy the visible pane only, without the prompt glyph.
     3. The progress rail tracks which section you are on.
     4. Sections reveal on scroll, unless the reader asked for reduced motion.
*/

(() => {
  "use strict";

  const OS_KEY = "recall-os";
  const VALID_OS = ["windows", "macos", "linux"];
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------------------------------------------------------- OS tabs */

  function readStoredOs() {
    try {
      const stored = window.localStorage.getItem(OS_KEY);
      if (stored && VALID_OS.includes(stored)) {
        return stored;
      }
    } catch {
      /* Private browsing or blocked storage. Fall through to detection. */
    }
    return detectOs();
  }

  function detectOs() {
    const ua = navigator.userAgent;
    if (/Windows/i.test(ua)) return "windows";
    if (/Mac OS X|Macintosh/i.test(ua)) return "macos";
    if (/Linux|X11/i.test(ua)) return "linux";
    return "windows";
  }

  function applyOs(os) {
    for (const group of document.querySelectorAll("[data-os-group]")) {
      const tabs = group.querySelectorAll("[data-os]");
      const panes = group.querySelectorAll("[data-os-pane]");

      // A block may not offer every OS. If the chosen one is absent, fall back
      // to the group's first pane rather than showing nothing at all.
      const offered = Array.from(panes).map((pane) => pane.dataset.osPane);
      const target = offered.includes(os) ? os : offered[0];

      for (const tab of tabs) {
        const selected = tab.dataset.os === target;
        tab.setAttribute("aria-selected", String(selected));
        tab.tabIndex = selected ? 0 : -1;
      }

      for (const pane of panes) {
        const selected = pane.dataset.osPane === target;
        pane.hidden = !selected;
        // A visible panel must be reachable by keyboard; its only content is a
        // non-focusable <pre>, so without this the command is unreachable to
        // anyone tabbing through the page.
        pane.tabIndex = selected ? 0 : -1;
      }
    }

    // The prompt glyph is a property of the shell, not of one block, so it
    // lives on the root. Blocks with a single OS-independent command (docker
    // ps) carry no tab strip, and would otherwise keep a POSIX prompt on a
    // Windows reader's screen while every neighbouring block switched.
    document.documentElement.dataset.shell = os === "windows" ? "powershell" : "posix";

    try {
      window.localStorage.setItem(OS_KEY, os);
    } catch {
      /* Not being able to remember the choice is not worth failing over. */
    }
  }

  function initOsTabs() {
    const groups = document.querySelectorAll("[data-os-group]");
    if (!groups.length) {
      document.documentElement.dataset.shell =
        readStoredOs() === "windows" ? "powershell" : "posix";
      return;
    }

    groups.forEach((group, groupIndex) => {
      const tabs = Array.from(group.querySelectorAll("[data-os]"));

      // Wire each tab to its panel. Without this the tab announces as a tab
      // controlling nothing, and the panel has no accessible name.
      for (const tab of tabs) {
        const os = tab.dataset.os;
        const pane = group.querySelector(`[data-os-pane="${os}"]`);
        if (!pane) continue;
        const tabId = `os-tab-${groupIndex}-${os}`;
        const paneId = `os-pane-${groupIndex}-${os}`;
        tab.id = tabId;
        pane.id = paneId;
        tab.setAttribute("aria-controls", paneId);
        pane.setAttribute("aria-labelledby", tabId);
      }

      group.addEventListener("click", (event) => {
        const tab = event.target.closest("[data-os]");
        if (tab && group.contains(tab)) applyOs(tab.dataset.os);
      });

      // Arrow-key navigation, which is what the tab role promises. Scoped to
      // the tabs themselves: the group is the whole terminal, so an unscoped
      // handler would swallow arrow keys aimed at the Copy button or at
      // scrolling a long command, and silently reset the whole page's OS.
      group.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
        const from = event.target.closest("[data-os]");
        if (!from || !group.contains(from)) return;
        const current = tabs.indexOf(from);
        if (current === -1) return;
        const step = event.key === "ArrowRight" ? 1 : -1;
        const next = tabs[(current + step + tabs.length) % tabs.length];
        event.preventDefault();
        applyOs(next.dataset.os);
        next.focus();
      });
    });

    applyOs(readStoredOs());
  }

  /* ------------------------------------------------------------------- copy */

  function textToCopy(terminal) {
    // Copy the visible pane only. A hidden PowerShell variant must not end up
    // on the clipboard of someone reading the macOS tab.
    const pane = terminal.querySelector("[data-os-pane]:not([hidden])") || terminal;
    const pre = pane.querySelector("pre");
    if (!pre) return "";

    // Read the line elements rather than pre.textContent, because the prompt
    // glyph is a generated ::before and must never be copied.
    const lines = pre.querySelectorAll(".line");
    if (lines.length) {
      return Array.from(lines)
        .map((line) => line.textContent.replace(/\s+$/, ""))
        .join("\n");
    }
    return pre.textContent.trim();
  }

  function legacyCopy(text) {
    // Clipboard access is refused over plain HTTP, when the document is not
    // focused, and in some embedded views. Copying from an off-screen textarea
    // keeps the prompt glyph out of the result, which selecting the <pre>
    // would not: Chrome and Safari include generated content in a selection.
    const scratch = document.createElement("textarea");
    scratch.value = text;
    scratch.setAttribute("readonly", "");
    scratch.setAttribute("aria-hidden", "true");
    scratch.style.cssText = "position:fixed;top:0;left:-9999px;opacity:0;";
    document.body.appendChild(scratch);
    scratch.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch {
      ok = false;
    }
    scratch.remove();
    return ok;
  }

  function initCopy() {
    for (const button of document.querySelectorAll(".copy-button")) {
      const terminal = button.closest(".terminal");
      if (!terminal) continue;

      button.addEventListener("click", async () => {
        const text = textToCopy(terminal);
        if (!text) return;

        const settle = (label, state) => {
          button.textContent = label;
          button.dataset.state = state;
          window.setTimeout(() => {
            button.textContent = "Copy";
            delete button.dataset.state;
          }, 1600);
        };

        try {
          await navigator.clipboard.writeText(text);
          settle("Copied", "done");
        } catch {
          const copied = legacyCopy(text);
          settle(copied ? "Copied" : "Copy failed", copied ? "done" : "fail");
        }
      });
    }
  }

  /* ------------------------------------------------------------------- rail */

  function initRail() {
    const rail = document.querySelector("[data-rail]");
    if (!rail) return;

    const links = Array.from(rail.querySelectorAll("a[href^='#']"));
    const steps = links
      .map((link) => document.getElementById(decodeURIComponent(link.hash.slice(1))))
      .filter(Boolean);
    if (!steps.length) return;

    const counter = rail.querySelector("[data-rail-progress]");
    // Troubleshooting lists categories, not steps. Let the page say which noun
    // it wants rather than relabelling its sections "Step 1 of 7".
    const noun = (counter && counter.dataset.railProgress) || "Step";
    const total = steps.length;
    let active = -1;

    const setActive = (index) => {
      if (index === active) return;
      active = index;
      links.forEach((link, i) => {
        link.dataset.active = String(i === index);
      });
      if (counter) {
        counter.textContent = `${noun} ${index + 1} of ${total}`;
      }
    };

    // The step whose top has most recently passed the reading line is the one
    // the reader is working on. Scanning on rAF-throttled scroll keeps this
    // correct on fast scrolls, which an IntersectionObserver ratio does not.
    let ticking = false;
    const update = () => {
      ticking = false;
      const line = window.innerHeight * 0.3;
      let index = 0;
      for (let i = 0; i < steps.length; i += 1) {
        if (steps[i].getBoundingClientRect().top <= line) index = i;
      }
      setActive(index);
    };

    window.addEventListener(
      "scroll",
      () => {
        if (!ticking) {
          ticking = true;
          window.requestAnimationFrame(update);
        }
      },
      { passive: true }
    );
    window.addEventListener("resize", update, { passive: true });
    update();
  }

  /* ---------------------------------------------------------------- reveals */

  function initReveals() {
    const targets = Array.from(document.querySelectorAll("[data-reveal]"));
    if (!targets.length) return;

    const showAll = () => {
      for (const target of targets) target.dataset.shown = "true";
    };

    if (reduceMotion || !("IntersectionObserver" in window)) {
      showAll();
      return;
    }

    // Arm everything inside a frame callback. An IntersectionObserver is
    // driven by the frame lifecycle, so where frames are not produced (a
    // background tab, a hidden view) it never delivers. Waiting for a real
    // frame means the stylesheet is only allowed to hide content in a context
    // that has already proved it can animate it back.
    requestAnimationFrame(() => {
      const observer = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            entry.target.dataset.shown = "true";
            observer.unobserve(entry.target);
          }
        },
        { rootMargin: "0px 0px -12% 0px", threshold: 0.05 }
      );

      // The failsafe must key on PROGRESS, not on the observer having spoken.
      // The spec queues an initial entry for every observed target, so a
      // "did the callback run" flag is set within a frame and would disarm
      // this before it can protect anything. What strands content is an
      // observer that runs and never reports a given target as intersecting:
      // a short block sitting entirely inside the bottom 12% margin at maximum
      // scroll can never satisfy the threshold. Anything still unshown when
      // the page has settled gets revealed.
      const sweep = () => {
        if (targets.some((target) => target.dataset.shown !== "true")) {
          const stranded = targets.filter((t) => {
            const rect = t.getBoundingClientRect();
            // Only rescue what is already on screen or above it; content
            // further down is simply not scrolled to yet.
            return t.dataset.shown !== "true" && rect.top < window.innerHeight;
          });
          for (const target of stranded) target.dataset.shown = "true";
        }
      };

      window.addEventListener("scroll", () => window.setTimeout(sweep, 250), {
        passive: true,
      });
      window.addEventListener("load", sweep);
      window.setTimeout(sweep, 1500);

      // Only now, with both the observer and the rescue path live, is it safe
      // to let the stylesheet hide anything.
      document.documentElement.classList.add("reveal-ready");
      for (const target of targets) observer.observe(target);
    });
  }

  /* --------------------------------------------------------------- masthead */

  function initMasthead() {
    const masthead = document.querySelector(".masthead");
    if (!masthead) return;

    let ticking = false;
    const update = () => {
      ticking = false;
      masthead.dataset.scrolled = String(window.scrollY > 8);
    };

    window.addEventListener(
      "scroll",
      () => {
        if (!ticking) {
          ticking = true;
          window.requestAnimationFrame(update);
        }
      },
      { passive: true }
    );
    update();
  }

  /* ------------------------------------------------------------------- boot */

  const boot = () => {
    // Each step is independent, and initReveals is the only one that can hide
    // anything. Isolating them means a failure in an earlier one cannot stop
    // the reveal path from arming, or from never arming, safely.
    for (const init of [initOsTabs, initCopy, initRail, initMasthead]) {
      try {
        init();
      } catch (error) {
        console.error("recall setup guide:", error);
      }
    }
    try {
      initReveals();
    } catch (error) {
      console.error("recall setup guide:", error);
      document.documentElement.classList.remove("reveal-ready");
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
