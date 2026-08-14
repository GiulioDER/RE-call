/* RE-call setup guide, behaviour layer.

   Four jobs, no dependencies:
     1. OS tabs switch every block on the page at once, and the choice sticks.
     2. Copy buttons copy the visible pane only, without the prompt glyph.
     3. The progress rail tracks which step you are on.
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
        pane.hidden = pane.dataset.osPane !== target;
      }

      // The prompt glyph differs between PowerShell and a POSIX shell.
      const terminal = group.closest(".terminal") || group;
      terminal.dataset.shell = target === "windows" ? "powershell" : "posix";
    }

    try {
      window.localStorage.setItem(OS_KEY, os);
    } catch {
      /* Not being able to remember the choice is not worth failing over. */
    }
  }

  function initOsTabs() {
    const groups = document.querySelectorAll("[data-os-group]");
    if (!groups.length) return;

    for (const group of groups) {
      const tabs = Array.from(group.querySelectorAll("[data-os]"));

      group.addEventListener("click", (event) => {
        const tab = event.target.closest("[data-os]");
        if (tab) applyOs(tab.dataset.os);
      });

      // Arrow-key navigation, which is what the tab role promises.
      group.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
        const current = tabs.findIndex((tab) => tab.getAttribute("aria-selected") === "true");
        if (current === -1) return;
        const step = event.key === "ArrowRight" ? 1 : -1;
        const next = tabs[(current + step + tabs.length) % tabs.length];
        event.preventDefault();
        applyOs(next.dataset.os);
        next.focus();
      });
    }

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
          // Clipboard access is refused over plain HTTP and in some embedded
          // views. Select the text so the reader can still copy it by hand.
          const pre = terminal.querySelector("[data-os-pane]:not([hidden]) pre, pre");
          if (pre) {
            const range = document.createRange();
            range.selectNodeContents(pre);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
          }
          settle("Select and copy", "fail");
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
    const total = steps.length;
    let active = -1;

    const setActive = (index) => {
      if (index === active) return;
      active = index;
      links.forEach((link, i) => {
        link.dataset.active = String(i === index);
      });
      if (counter) {
        counter.textContent = `Step ${index + 1} of ${total}`;
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
    const targets = document.querySelectorAll("[data-reveal]");
    if (!targets.length) return;

    const showAll = () => {
      for (const target of targets) target.dataset.shown = "true";
    };

    if (reduceMotion || !("IntersectionObserver" in window)) {
      showAll();
      return;
    }

    // Arm the whole thing inside a frame callback. An IntersectionObserver is
    // driven by the frame lifecycle, so where frames are not being produced
    // (a background tab, a hidden view) it never delivers. Waiting for a real
    // frame means the stylesheet is only ever allowed to hide content in a
    // context that has already proved it can animate it back.
    requestAnimationFrame(() => {
      document.documentElement.classList.add("reveal-ready");

      let delivered = false;
      const observer = new IntersectionObserver(
        (entries) => {
          delivered = true;
          for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            entry.target.dataset.shown = "true";
            observer.unobserve(entry.target);
          }
        },
        { rootMargin: "0px 0px -12% 0px", threshold: 0.05 }
      );

      for (const target of targets) observer.observe(target);

      // Second line of defence, for an observer that runs but never fires.
      // Hidden install instructions are worse than a lost animation.
      window.setTimeout(() => {
        if (!delivered) {
          observer.disconnect();
          showAll();
        }
      }, 1500);
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
    initOsTabs();
    initCopy();
    initRail();
    initReveals();
    initMasthead();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
