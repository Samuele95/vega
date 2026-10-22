// The tour: the views in order, reached by the tabs, the index, the number keys and the next button;
// the start view (0, esc); and views off the tour (the code), opened by their own key and left with
// esc back to where the reader was. Words come from the caller's say(key, params):
//   view.<id>              the label of a view, "start" and the off-tour views included
//   tour.index.<off id>    the index entry of an off-tour view
//   tour.next              the next button ({label})
//   tour.progress          the progress caption ({i}, {n}, {label})
// The DOM it drives: .view[data-view], #crumb-here, [data-progress], #tabs, #index-list,
// #index-overlay, #open-index, #next.
import { $, el, restart, store } from "./dom.js";
import { pad2 } from "./fmt.js";
import { S } from "./state.js";

const START = "start";

export class Tour {
  // views: the tour's view ids, in order; offTour: [{id, key, open}]; aliases: {old id: id};
  // onShow(id) runs after a view is shown; hotkeys: {key: fn(event)}; viewKeys: {view id: fn(event)}.
  constructor({ views, offTour = [], aliases = {}, say, onShow = () => {}, hotkeys = {}, viewKeys = {} }) {
    this.ids = views.slice();
    this.off = offTour.slice();
    this.opts = { aliases, say, onShow, hotkeys, viewKeys };
    this.at = START;
    this.backTo = START;
  }

  get current() { return this.at; }
  get views() { return this.ids.slice(); }
  label(id) { return this.opts.say("view." + id); }
  isOff(id) { return this.off.some((o) => o.id === id); }

  // The tabs, the index sheet and the keys.
  build() {
    const say = this.opts.say;
    this.ids.forEach((id, i) => {
      $("tabs").append(Object.assign(el("button", { "data-view": id }, el("span", { class: "k" }, String(i + 1)), this.label(id)), { onclick: () => this.show(id) }));
      $("index-list").append(Object.assign(el("button", {}, el("span", { class: "k" }, String(i + 1)), this.label(id)), { onclick: () => { this.closeIndex(); this.show(id); } }));
    });
    $("index-list").append(Object.assign(el("button", {}, el("span", { class: "k" }, "0"), this.label(START)), { onclick: () => { this.closeIndex(); this.show(START); } }));
    this.off.forEach((o) => $("index-list").append(el("div", { class: "rule", style: "margin:12px 0" }),
      Object.assign(el("button", { class: "muted" }, el("span", { class: "k" }, o.key), say("tour.index." + o.id)), { onclick: () => { this.closeIndex(); o.open(); } })));
    $("open-index").onclick = () => this.openIndex();
    $("index-overlay").onclick = (e) => { if (e.target.id === "index-overlay") this.closeIndex(); };
    document.addEventListener("keydown", (e) => this.onKey(e));
  }

  openIndex() { $("index-overlay").classList.add("open"); }
  closeIndex() { $("index-overlay").classList.remove("open"); }
  indexOpen() { return $("index-overlay").classList.contains("open"); }

  progressInto(host) {
    host.replaceChildren();
    this.ids.forEach((id) => host.append(el("span", { class: (S.seen.has(id) ? "seen " : "") + (id === this.at ? "here" : "") })));
    const i = this.ids.indexOf(this.at);
    if (i >= 0) host.append(el("em", { class: "caps" }, this.opts.say("tour.progress", { i: pad2(i + 1), n: pad2(this.ids.length), label: this.label(this.at) })));
  }

  show(id, push = true) {
    id = this.opts.aliases[id] || id;
    if (![START, ...this.off.map((o) => o.id), ...this.ids].includes(id)) id = START;
    if (this.isOff(id) && this.at !== id) this.backTo = this.at;
    const prev = document.querySelector(`.view[data-view="${this.at}"]`);
    const next = document.querySelector(`.view[data-view="${id}"]`);
    if (prev && prev !== next) prev.hidden = true;
    this.at = id;
    if (id !== START) { S.seen.add(id); store.set("seen", [...S.seen].join(",")); }
    next.hidden = false; restart(next); next.scrollTop = 0;
    $("crumb-here").textContent = this.label(id);
    document.querySelectorAll("[data-progress]").forEach((h) => this.progressInto(h));
    [...$("tabs").children].forEach((b) => b.classList.toggle("on", b.dataset.view === id));
    if (push) history.replaceState(null, "", id === START ? location.pathname : "#" + id);
    this.nextButton(id);
    this.opts.onShow(id);
  }

  nextButton(id) {
    const i = this.ids.indexOf(id), nx = this.ids[i + 1], btn = $("next");
    if (id !== START && i >= 0 && nx) {
      btn.textContent = this.opts.say("tour.next", { label: this.label(nx) });
      btn.onclick = () => this.show(nx); btn.classList.add("show");
    } else btn.classList.remove("show");
  }

  // Leaves an off-tour view for the view it was opened from.
  back() { this.show(this.backTo && this.backTo !== this.at ? this.backTo : START); }

  // Toggles an off-tour view: open it, or go back when it is the current one.
  toggleOff(o) {
    this.closeIndex();
    if (this.at === o.id) this.back(); else o.open();
  }

  onKey(e) {
    if (e.target.closest && e.target.closest("input,textarea")) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const off = this.off.find((o) => o.key === e.key), n = /^[1-9]$/.test(e.key) ? +e.key : 0;
    if (e.key === "Escape") { if (this.indexOpen()) this.closeIndex(); else if (this.isOff(this.at)) this.back(); else this.show(START); }
    else if (off) this.toggleOff(off);
    else if (n && n <= this.ids.length) this.show(this.ids[n - 1]);
    else if (e.key === "0") this.show(START);
    else if (e.key === "i") this.openIndex();
    else if (this.opts.hotkeys[e.key]) this.opts.hotkeys[e.key](e);
    else if (this.opts.viewKeys[this.at]) this.opts.viewKeys[this.at](e);
  }
}
