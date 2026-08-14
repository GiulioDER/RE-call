document.addEventListener("DOMContentLoaded", () => {
  for (const button of document.querySelectorAll("[data-copy-target]")) {
    button.addEventListener("click", async () => {
      const selector = button.getAttribute("data-copy-target");
      const source = selector ? document.querySelector(selector) : null;
      if (!source) {
        return;
      }
      const text = source.textContent || "";
      try {
        await navigator.clipboard.writeText(text.trim());
        const original = button.textContent;
        button.textContent = "Copied";
        window.setTimeout(() => {
          button.textContent = original;
        }, 1200);
      } catch {
        button.textContent = "Copy failed";
        window.setTimeout(() => {
          button.textContent = "Copy";
        }, 1200);
      }
    });
  }
});
