/* vega-clock — one frame rate for the site's drawing. Every page runs several canvas / WebGL loops (galaxy, scenes, sky,
   readouts, the reel's light); on a high-refresh display (120–240 Hz) each would run 2–4× per 60 Hz frame, and their
   script time alone outgrew a 240 Hz frame's 4.2 ms budget (G8: 2.6–5.4 ms measured at DPR 1.6). requestAnimationFrame
   callbacks are therefore batched and run at most once per ~13.3 ms (≈60 fps on 120 / 240 Hz, 72 fps on 144 Hz,
   every frame at 60–75 Hz). Native scrolling, CSS transitions and Web Animations keep the display's own rate.
   Classic script, first in <body> on every page (before any module asks for a frame). */
(() => {
  if (window.VegaClock) return;
  const native = window.requestAnimationFrame.bind(window), MIN = 1000 / 60 * 0.8;
  let queue = new Map(), id = 0, pending = false, last = -1e9;
  const pump = (ts) => {
    if (ts - last < MIN) { native(pump); return; } // too soon after the last drawn frame: wait for the next vsync
    pending = false; last = ts;
    const q = queue; queue = new Map();
    q.forEach(cb => { try { cb(ts); } catch (e) { setTimeout(() => { throw e; }); } }); // one failing loop never stops the others (as natively)
  };
  window.requestAnimationFrame = (cb) => { const h = ++id; queue.set(h, cb); if (!pending) { pending = true; native(pump); } return h; };
  window.cancelAnimationFrame = (h) => { queue.delete(h); };
  window.VegaClock = { MIN };
})();
