// Disclosures: a `.disclose` button (aria-expanded, data-for="<id>") opens and closes the `.reveal`
// with that id. A module that draws inside a disclosure registers with onDisclose(id, fn), and fn
// runs each time the disclosure opens.
import { $ } from "./dom.js";
import { LISTENERS } from "./state.js";

export function onDisclose(id, fn) {
  if (!LISTENERS.disclose.has(id)) LISTENERS.disclose.set(id, []);
  LISTENERS.disclose.get(id).push(fn);
}

function toggle(b) {
  const open = b.getAttribute("aria-expanded") !== "true", id = b.dataset.for;
  b.setAttribute("aria-expanded", String(open));
  $(id).classList.toggle("open", open);
  if (open) (LISTENERS.disclose.get(id) || []).forEach((fn) => fn(id));
}

// Binds every disclosure button under root (the document by default).
export function bindDisclosures(root = document) {
  root.querySelectorAll(".disclose").forEach((b) => b.addEventListener("click", () => toggle(b)));
}

export function openDisclose(id) {
  const b = document.querySelector(`.disclose[data-for="${id}"]`);
  if (b && b.getAttribute("aria-expanded") !== "true") b.click();
}
