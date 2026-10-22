/* vega-audio — the site's synthesized sound (spec §2.7). One AudioContext, created or resumed only
   inside a user gesture (ENTER, the SOUND toggle, or the first gesture of a later page load when the
   visitor left sound on). Layers: a low drone (two detuned oscillators through a slow low-pass) whose
   gain and cutoff follow camera speed, a soft tick on stop change, a very quiet click on HUD hover/focus.
   Off by default; the choice persists in localStorage 'vega-sound'; muted while the tab is hidden.
   Every level change is a ramp of ≥150 ms, so nothing clicks or pops.
   Events: window 'vegasound' { detail: { on } } whenever the preference changes. */
(() => {
  if (window.VegaAudio) return;
  const KEY = 'vega-sound', LEVEL = 0.6, FADE_IN = 0.6, FADE_OUT = 0.25;
  let pref = false;
  try { pref = localStorage.getItem(KEY) === 'on'; } catch (e) {}
  let ctx = null, master = null, drone = null, filter = null, noise = null, lastClick = 0, offTimer = 0;

  const AC = () => window.AudioContext || window.webkitAudioContext;
  const now = () => ctx.currentTime;
  const ramp = (param, to, secs) => { const t = now(); param.cancelScheduledValues(t); param.setValueAtTime(param.value, t); param.linearRampToValueAtTime(to, t + secs); };

  function build() {
    const C = AC(); if (!C) return false;
    ctx = new C();
    master = ctx.createGain(); master.gain.value = 0; master.connect(ctx.destination);
    // drone: two detuned saws, low-passed, the cutoff drifting on a very slow LFO
    filter = ctx.createBiquadFilter(); filter.type = 'lowpass'; filter.frequency.value = 240; filter.Q.value = 0.8;
    drone = ctx.createGain(); drone.gain.value = 0.07;
    filter.connect(drone); drone.connect(master);
    [55, 55 * Math.pow(2, 9 / 1200)].forEach((f) => { const o = ctx.createOscillator(); o.type = 'sawtooth'; o.frequency.value = f; o.connect(filter); o.start(); });
    const lfo = ctx.createOscillator(), depth = ctx.createGain(); lfo.frequency.value = 0.045; depth.gain.value = 80; lfo.connect(depth); depth.connect(filter.frequency); lfo.start();
    // a short noise buffer for the HUD click
    noise = ctx.createBuffer(1, Math.round(ctx.sampleRate * 0.03), ctx.sampleRate);
    const d = noise.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    return true;
  }

  function start() {
    clearTimeout(offTimer);
    if (!ctx && !build()) return;
    const p = ctx.state === 'running' ? Promise.resolve() : ctx.resume();
    Promise.resolve(p).then(() => { if (pref && !document.hidden) ramp(master.gain, LEVEL, FADE_IN); }).catch(() => {});
  }
  function stop() {
    if (!ctx) return;
    ramp(master.gain, 0, FADE_OUT);
    clearTimeout(offTimer);
    offTimer = setTimeout(() => { if (ctx && (!pref || document.hidden)) ctx.suspend().catch(() => {}); }, FADE_OUT * 1000 + 80);
  }
  const running = () => !!ctx && ctx.state === 'running' && pref && !document.hidden;

  function set(on) {
    pref = !!on;
    try { localStorage.setItem(KEY, pref ? 'on' : 'off'); } catch (e) {}
    if (pref) start(); else stop();
    disarm();
    dispatchEvent(new CustomEvent('vegasound', { detail: { on: pref } }));
  }

  // a visitor who left sound on: wait for this page's first gesture (no context before it)
  const GESTURES = ['pointerdown', 'keydown', 'touchend'];
  const onGesture = (e) => { if (e.type === 'keydown' && e.key === 'Escape') return; if (pref) start(); disarm(); };
  let armed = false;
  function arm() { if (armed || !pref) return; armed = true; GESTURES.forEach(g => addEventListener(g, onGesture, true)); }
  function disarm() { if (!armed) return; armed = false; GESTURES.forEach(g => removeEventListener(g, onGesture, true)); }
  arm();

  document.addEventListener('visibilitychange', () => {
    if (!ctx) return;
    if (document.hidden) stop(); else if (pref) start();
  });

  window.VegaAudio = {
    get on() { return pref; },
    set,
    toggle() { set(!pref); },
    // ENTER is a gesture: start only if the visitor already chose sound
    enter() { if (pref) { start(); disarm(); } },
    // 0..1 camera speed → drone level and brightness
    motion(v) {
      if (!running()) return;
      v = Math.max(0, Math.min(1, v || 0));
      const t = now();
      filter.frequency.setTargetAtTime(240 + v * 680, t, 0.25);
      drone.gain.setTargetAtTime(0.07 + v * 0.06, t, 0.25);
    },
    tick() {
      if (!running()) return;
      const t = now(), o = ctx.createOscillator(), g = ctx.createGain();
      o.type = 'sine'; o.frequency.setValueAtTime(1180, t); o.frequency.exponentialRampToValueAtTime(820, t + 0.12);
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.05, t + 0.006); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.16);
      o.connect(g); g.connect(master); o.start(t); o.stop(t + 0.18);
    },
    click() {
      if (!running()) return;
      const ms = performance.now(); if (ms - lastClick < 70) return; lastClick = ms;
      const t = now(), s = ctx.createBufferSource(), bp = ctx.createBiquadFilter(), g = ctx.createGain();
      s.buffer = noise; bp.type = 'bandpass'; bp.frequency.value = 3200; bp.Q.value = 1.2;
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.012, t + 0.003); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.028);
      s.connect(bp); bp.connect(g); g.connect(master); s.start(t); s.stop(t + 0.03);
    },
    state() { return ctx ? ctx.state : 'none'; }
  };
})();
