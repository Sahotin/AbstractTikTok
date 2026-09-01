/**
 * Apple 风格布局与主题增强
 * 让右上角 Light / Dark / Auto 下拉选择真正影响页面主题。
 */
(function () {
  const STORAGE_KEY = "abstract-tiktok-apple-theme";
  const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  const normalizeTheme = (value) => {
    const theme = String(value || "").trim().toLowerCase();
    if (theme === "dark") return "dark";
    if (theme === "auto" || theme === "system") return "auto";
    return "light";
  };

  const getPreferredTheme = () => normalizeTheme(localStorage.getItem(STORAGE_KEY) || "light");

  const resolveTheme = (theme) => {
    const normalized = normalizeTheme(theme);
    if (normalized === "auto") {
      return media && media.matches ? "dark" : "light";
    }
    return normalized;
  };

  const applyTheme = (theme, persist) => {
    const normalized = normalizeTheme(theme);
    const resolved = resolveTheme(normalized);

    document.documentElement.classList.toggle("dark", resolved === "dark");
    document.documentElement.dataset.appleTheme = normalized;
    document.documentElement.style.colorScheme = resolved;

    if (persist) {
      localStorage.setItem(STORAGE_KEY, normalized);
    }
  };

  const applyLayout = () => {
    const root = document.getElementById("root");
    if (!root) return;

    const shell = root.querySelector(".cyber-grid, .app-shell");
    if (shell && !shell.classList.contains("app-shell")) {
      shell.classList.add("app-shell");
    }
  };

  /** 解除 Tailwind h-screen / overflow-hidden，使配置区与控制台可滚动 */
  const enablePageScroll = () => {
    document.documentElement.style.overflowY = "auto";
    document.body.style.overflowY = "auto";
    document.body.style.height = "auto";

    const root = document.getElementById("root");
    if (!root) return;

    const shell = root.querySelector(".app-shell, .cyber-grid");
    if (!shell) return;

    shell.classList.remove("h-screen", "overflow-hidden");
    shell.style.setProperty("height", "auto", "important");
    shell.style.setProperty("min-height", "100vh", "important");
    shell.style.setProperty("overflow-y", "auto", "important");
    shell.style.setProperty("overflow-x", "hidden", "important");

    const mainWrap = shell.querySelector(":scope > .flex-1.flex.flex-col");
    if (mainWrap) {
      mainWrap.classList.remove("overflow-hidden");
      mainWrap.style.setProperty("overflow-y", "auto", "important");
      mainWrap.style.setProperty("max-height", "none", "important");
    }

    shell.querySelectorAll("main.overflow-hidden").forEach((el) => {
      el.classList.remove("overflow-hidden");
      el.style.setProperty("overflow", "visible", "important");
    });
  };

  const apply = () => {
    applyTheme(getPreferredTheme(), false);
    applyLayout();
    enablePageScroll();
  };

  apply();

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", apply);
  }

  document.addEventListener(
    "click",
    (event) => {
      const option = event.target && event.target.closest ? event.target.closest('[role="option"], [data-radix-collection-item]') : null;
      if (!option) return;

      const text = (option.textContent || "").trim().toLowerCase();
      if (text.includes("dark")) {
        applyTheme("dark", true);
      } else if (text.includes("auto") || text.includes("system")) {
        applyTheme("auto", true);
      } else if (text.includes("light")) {
        applyTheme("light", true);
      }
    },
    true
  );

  if (media) {
    const handleSystemTheme = () => {
      if (getPreferredTheme() === "auto") {
        applyTheme("auto", false);
      }
    };

    if (media.addEventListener) {
      media.addEventListener("change", handleSystemTheme);
    } else if (media.addListener) {
      media.addListener(handleSystemTheme);
    }
  }

  const target = document.getElementById("root") || document.body;
  if (target) {
    new MutationObserver(applyLayout).observe(target, { childList: true, subtree: true });
  }
})();
