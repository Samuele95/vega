// The two themes, night (the default) and day: the stored choice, the switch, and the colours read
// from the CSS tokens of the current theme (cached until the theme changes).
import { css, store } from "./dom.js";
import { CACHE, LISTENERS } from "./state.js";

const THEMES = Object.freeze(["night", "day"]);

export const theme = () => document.documentElement.dataset.theme;

// Applies the theme remembered by this browser, if any; night otherwise (the markup's default).
export function applyStoredTheme() {
  const t = store.get("theme");
  if (THEMES.includes(t)) document.documentElement.dataset.theme = t;
}

// Calls fn(theme) after every switch, in registration order.
export function onTheme(fn) {
  LISTENERS.theme.push(fn);
}

export function toggleTheme() {
  const t = theme() === "night" ? "day" : "night";
  document.documentElement.dataset.theme = t; store.set("theme", t);
  CACHE.colours = null;
  LISTENERS.theme.forEach((fn) => fn(t));
}

// The theme's colours, as the canvas drawings use them.
export function colours() {
  if (!CACHE.colours) {
    CACHE.colours = Object.freeze({
      acc: css("--accent"), neg: css("--neg"), mut: css("--muted2"), fg: css("--fg"), fg2: css("--fg2"),
      muted: css("--muted"), alert: css("--alert"), bg: css("--bg"), grid: css("--grid"), lane: css("--lane"),
      line: css("--line"), lineHi: css("--line-hi"),
    });
  }
  return CACHE.colours;
}
