/* vega-galaxy — the universe behind the pages (galaxy plan, G1). Hand-written WebGL, no library: a spiral galaxy of
   point stars in VEGA's palette (--fg, --acc, --cy, --dim), turning slowly, with the VEGA star at its core drawn after
   the mark — a glow and four spikes. It sits between the far sky (vega-sky-lab) and a page's own scene.
   View: el.drive = () => ({ dist, yaw, pitch, anchor: [x, y] }) is read every frame (the page maps its scroll camera to a
   path through the galaxy; `anchor` is the screen point where the core sits), and the view eases toward it. Without a
   driver it rests on a wide view. el.exposure (0..1) scales its light.
   el.bodies = () => list — planets drawn as lit 3D spheres at screen spots a page already projects (index's page stars):
     [{ x, y (css px), r (px), L: [x, y, z] (direction to VEGA, screen space, y up, z toward the viewer), z (depth), i, a }],
     list.coreZ = VEGA's depth (bodies farther than it draw behind the star).
   el.cam = { eye, fwd, up } — a free camera in galaxy space (flight); it replaces drive while set. el.system = true draws
     the VEGA system in galaxy space: six planets on their orbits (el.planets(t) gives their positions) and the rings.
   el.tick(dt, now) runs at the start of each frame, el.onframe(frame) after it ({ eye, fwd, PV, W, H }: project labels).
   el.home = k (0..5) with el.homeU = () => 0..1 — an inner page: the camera beside planet k, orbiting with the page's scroll,
     VEGA pinned at el.homeAt (NDC, default upper-left [-0.62, 0.6])
     (set el.system = true to see the system's bodies and rings).
   el.warp (0..1) — the jump: the camera runs forward along its view and every star stretches into a streak behind
     it (el.warpAt = [x, y] css px moves the vanishing point there).
   Quality: el.q (0 full … 3 lowest) steps down on its own when frames run long (see loop); software GL starts at 2.
   Reduced motion: no drift, no twinkle, the view cuts. No WebGL: the element hides itself and the page keeps its
   canvas 2D sky. It pauses while hidden. */
(() => {
  if (customElements.get('vega-galaxy')) return;
  const RM = matchMedia('(prefers-reduced-motion: reduce)');
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  const STAR_VS = `
attribute vec3 a_pos; attribute vec3 a_col; attribute float a_size; attribute float a_seed; attribute float a_alpha;
uniform mat4 u_mvp; uniform vec2 u_anchor; uniform float u_px; uniform float u_time; uniform float u_tw; uniform float u_exp;
varying vec3 v_col; varying float v_a; varying float v_spk;
void main() {
  vec4 p = u_mvp * vec4(a_pos, 1.0);
  p.xy += u_anchor * p.w;
  gl_Position = p;
  float w = max(p.w, 1.0);
  float tw = 1.0 - u_tw * 0.35 * (0.5 + 0.5 * sin(u_time * (0.6 + a_seed * 1.7) + a_seed * 40.0));
  float cloud = step(20.0, a_size); // the glow points: they draw the spiral from afar and thin out up close
  v_spk = step(2.9, a_size) * (1.0 - cloud); // the brightest stars carry the mark's four spikes
  float s = a_size * u_px * clamp(900.0 / w, 0.35, mix(3.2, 1.1, cloud)) * (1.0 + v_spk * 4.0);
  gl_PointSize = max(s, 1.0);
  v_a = a_alpha * tw * u_exp * clamp(s / max(gl_PointSize, 1.0), 0.35, 1.0) * clamp(5200.0 / w, 0.0, 1.0) * mix(1.0, clamp(w / 1600.0, 0.05, 1.0), cloud);
  v_col = a_col;
}`;
  const STAR_FS = `
precision mediump float;
varying vec3 v_col; varying float v_a; varying float v_spk;
void main() {
  vec2 d = gl_PointCoord - 0.5; float r = length(d) * 2.0;
  float a = smoothstep(1.0, 0.0, r); a *= a;
  if (v_spk > 0.5) { float c = smoothstep(0.2, 0.0, r); a = c * c + (exp(-abs(d.x) * 90.0) * exp(-abs(d.y) * 6.0) + exp(-abs(d.y) * 90.0) * exp(-abs(d.x) * 7.0)) * 0.75; }
  gl_FragColor = vec4(v_col * a * v_a, a * v_a);
}`;
  const NOISE = `
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + 1.0), f.x), f.y); }`;
  // the VEGA star: a screen-space quad around the projected core, in canvas px. The star (glow, disc, four diffraction
  // spikes, a living corona, an anamorphic flare) and — where the page asks (u_L > 0, index's hero) — VEGA's mark drawn
  // in light at the mark's own geometry: the two tapered spikes (27 / 23 of 64), their tip dots, the diagonal rays and
  // satellites, the core; u_in replays the mark's draw-in, u_k hands over between the mark and the pure star
  const CORE_VS = `
attribute vec2 a_q; uniform vec2 u_c; uniform vec2 u_r;
varying vec2 v_q;
void main() { v_q = a_q; gl_Position = vec4(u_c + a_q * u_r, 0.0, 1.0); }`;
  const CORE_FS = `
precision highp float;
varying vec2 v_q; uniform vec2 u_R; uniform float u_S; uniform float u_L; uniform float u_k; uniform float u_in;
uniform float u_t; uniform float u_pulse; uniform float u_exp; uniform float u_flare; uniform vec3 u_fg; uniform vec3 u_acc;
` + NOISE + `
float seg(vec2 p, vec2 a, vec2 b) { vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0); return length(pa - ba * h); }
float pop(float x) { return x * (1.0 + 0.9 * sin(clamp(x, 0.0, 1.0) * 3.14159) * step(x, 0.999)); }
void main() {
  vec2 p = v_q * u_R; float r = length(p), S = u_S;
  float star = 1.0 - u_k * 0.85;
  float glow = exp(-r / (S / 5.5)) * 0.9 + exp(-r / (S / 18.0)) * 0.8;
  float disc = clamp((S * 0.035 - r) / 1.2 + 0.5, 0.0, 1.0) * star;
  float sx = exp(-abs(p.y) / max(S / 220.0, 0.6)) * exp(-abs(p.x) / (S / 3.2)), sy = exp(-abs(p.x) / max(S / 220.0, 0.6)) * exp(-abs(p.y) / (S / 3.2));
  float spikes = (sx + sy) * (0.8 + 0.4 * u_pulse) * star;
  float ang = atan(p.y, p.x), rs = r / S;
  float n = noise(vec2(ang * 5.0, rs * 4.0 - u_t * 0.35)) * 0.6 + noise(vec2(ang * 11.0 + 3.0, rs * 9.0 - u_t * 0.6)) * 0.4;
  float corona = exp(-r / (S * 0.3)) * (0.25 + 0.9 * n * n) * 0.55 * (0.85 + 0.3 * u_pulse);
  float fl = exp(-abs(p.y) / (1.0 + S * 0.006)) * exp(-abs(p.x) / (S * 2.2)) * u_flare;
  vec3 col = u_acc * (glow + corona + fl * 0.7) + u_fg * (disc + spikes * 0.72);
  float al = glow + disc + spikes + corona + fl * 0.7;
  if (u_L > 0.0 && u_k > 0.001) {
    float s = u_L / 27.0, aa = 1.0 / s; vec2 m = abs(p / s);
    float i1 = smoothstep(0.0, 0.3, u_in), iv = smoothstep(0.15, 0.5, u_in), ih = smoothstep(0.23, 0.58, u_in), it = smoothstep(0.38, 0.62, u_in), il = smoothstep(0.6, 0.8, u_in), ia = smoothstep(0.74, 0.96, u_in);
    float vy = 27.0 * iv, hx = 23.0 * ih;
    float spV = clamp(0.5 - (m.x - 1.5 * max(0.0, 1.0 - m.y / max(vy, 0.01))) / aa, 0.0, 1.0) * step(m.y, vy);
    float spH = clamp(0.5 - (m.y - 1.4 * max(0.0, 1.0 - m.x / max(hx, 0.01))) / aa, 0.0, 1.0) * step(m.x, hx);
    float rt = 2.8 * pop(it), dt = min(length(m - vec2(0.0, 27.0)), length(m - vec2(23.0, 0.0)));
    float tips = clamp((rt - dt) / aa + 0.5, 0.0, 1.0);
    float core = clamp((5.2 * pop(i1) - length(m)) / aa + 0.5, 0.0, 1.0);
    float ray = clamp(0.5 - (seg(m, vec2(4.2), vec2(4.2) + vec2(9.8) * il) - 0.5) / aa, 0.0, 1.0) * step(0.001, il);
    float sat = clamp((2.0 * pop(ia) - length(m - vec2(14.0))) / aa + 0.5, 0.0, 1.0);
    float halo = exp(-r / (9.0 * s)) * 0.3 * i1 + exp(-r / (2.5 * s)) * 0.22 * i1;
    float fgA = max(max(spV, spH), max(tips, core)), acA = max(ray, sat) * 0.95;
    col += (u_fg * fgA * 0.82 + u_acc * (acA * 0.8 + halo * (0.7 + 0.5 * u_pulse))) * u_k;
    al += (max(fgA, acA) + halo) * u_k;
  }
  gl_FragColor = vec4(col * u_exp, clamp(al, 0.0, 1.0) * u_exp);
}`;
  // post: a bloom (bright pass at a quarter of the size, two separable blurs) and the composite: soft shoulder, grain
  const FS_VS = `
attribute vec2 a_q; varying vec2 v_uv;
void main() { v_uv = a_q * 0.5 + 0.5; gl_Position = vec4(a_q, 0.0, 1.0); }`;
  const BRIGHT_FS = `
precision mediump float;
varying vec2 v_uv; uniform sampler2D u_tex; uniform vec2 u_px; uniform float u_th;
void main() {
  vec3 c = (texture2D(u_tex, v_uv + u_px * vec2(-1.0, -1.0)).rgb + texture2D(u_tex, v_uv + u_px * vec2(1.0, -1.0)).rgb
          + texture2D(u_tex, v_uv + u_px * vec2(-1.0, 1.0)).rgb + texture2D(u_tex, v_uv + u_px * vec2(1.0, 1.0)).rgb) * 0.25;
  float l = max(c.r, max(c.g, c.b));
  gl_FragColor = vec4(c * smoothstep(u_th, u_th + 0.35, l), 1.0);
}`;
  const BLUR_FS = `
precision mediump float;
varying vec2 v_uv; uniform sampler2D u_tex; uniform vec2 u_dir;
void main() {
  vec3 c = texture2D(u_tex, v_uv).rgb * 0.227;
  c += (texture2D(u_tex, v_uv + u_dir * 1.385).rgb + texture2D(u_tex, v_uv - u_dir * 1.385).rgb) * 0.316;
  c += (texture2D(u_tex, v_uv + u_dir * 3.231).rgb + texture2D(u_tex, v_uv - u_dir * 3.231).rgb) * 0.070;
  gl_FragColor = vec4(c, 1.0);
}`;
  const COMP_FS = `
precision mediump float;
varying vec2 v_uv; uniform sampler2D u_scene; uniform sampler2D u_b1; uniform sampler2D u_b2; uniform vec2 u_res; uniform float u_t; uniform float u_bk; uniform float u_grain;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main() {
  vec4 sc = texture2D(u_scene, v_uv);
  vec3 b = texture2D(u_b1, v_uv).rgb * 0.75 + texture2D(u_b2, v_uv).rgb * 0.9;
  vec3 c = sc.rgb + b * u_bk;
  float m = max(c.r, max(c.g, c.b)); c = c / (1.0 + max(m - 0.85, 0.0) * 0.6); // a soft shoulder: highlights roll off, not clip
  float a = clamp(max(sc.a, max(b.r, max(b.g, b.b)) * u_bk), 0.0, 1.0);
  float g = (hash(v_uv * u_res + fract(u_t * 7.13) * 91.7) - 0.5) * u_grain;
  c = max(c + g * (0.25 + a), 0.0);
  gl_FragColor = vec4(c, max(a, max(c.r, max(c.g, c.b)) * 0.0));
}`;
  // a planet: a sphere impostor on a screen quad — per-pixel normal, lit from VEGA's direction (u_L, screen space:
  // x right, y up, z toward the viewer), banded surface turning with u_spin, a soft terminator and an atmosphere rim
  const BODY_VS = `
attribute vec2 a_q; uniform vec2 u_c; uniform vec2 u_r;
varying vec2 v_q;
void main() { v_q = a_q * 1.4; gl_Position = vec4(u_c + a_q * u_r * 1.4, 0.0, 1.0); }`;
  const BODY_FS = `
precision highp float;
varying vec2 v_q; uniform vec3 u_L; uniform vec3 u_base; uniform vec3 u_band; uniform vec3 u_rim;
uniform float u_exp; uniform float u_seed; uniform float u_spin; uniform float u_alpha; uniform float u_aa; uniform float u_t;
` + NOISE + `
void main() {
  vec2 q = v_q; float rr = length(q);
  if (rr > 1.0) { // the atmosphere, outside the disc, on the lit side
    float h = clamp(1.0 - (rr - 1.0) / 0.4, 0.0, 1.0); h = h * h * h;
    float lit = clamp(dot(vec3(q / rr, 0.0), u_L) * 0.6 + 0.45, 0.0, 1.0);
    lit = max(lit, clamp(-u_L.z, 0.0, 1.0) * 1.3); // backlit: the atmosphere glows all round (forward scatter)
    float a = h * 0.5 * lit * u_alpha * u_exp;
    gl_FragColor = vec4(u_rim * a, a * 0.5); return;
  }
  vec3 n = vec3(q, sqrt(1.0 - rr * rr));
  float lon = atan(n.x, n.z) + u_spin, lat = n.y;
  vec2 uv = vec2(lon * 2.0, lat * 6.0) + u_seed;
  float fb = noise(uv) * 0.5 + noise(uv * 2.03 + 3.1) * 0.25 + noise(uv * 4.11 + 7.7) * 0.125 + noise(uv * 8.37 + 1.3) * 0.0625; // fbm: detail up close
  float b = sin(lat * 9.0 + fb * 3.2) * 0.5 + 0.5;
  float fine = noise(vec2(lon * 9.0, lat * 26.0) + u_seed * 3.0) * 0.6 + noise(vec2(lon * 21.0, lat * 60.0) + u_seed) * 0.4;
  vec3 alb = mix(u_base, u_band, clamp(b * 0.6 + fine * 0.28, 0.0, 1.0));
  float d = dot(n, u_L);
  float diff = smoothstep(-0.12, 0.3, d) * (0.3 + 0.7 * max(d, 0.0));
  float rim = pow(1.0 - n.z, 2.2) * smoothstep(-0.25, 0.35, d);
  float back = pow(1.0 - n.z, 3.5) * clamp(-u_L.z, 0.0, 1.0) * 1.6; // a backlit limb: the thin bright ring of an eclipse
  // a cloud deck turning faster than the surface, and the key light's glint
  vec2 cu = vec2((lon + u_t * 0.035) * 3.0, lat * 7.0) + u_seed * 1.7;
  float cl = smoothstep(0.55, 0.85, noise(cu) * 0.6 + noise(cu * 2.3 + 5.0) * 0.3 + noise(cu * 5.1) * 0.1) * 0.7;
  alb = mix(alb, vec3(0.92, 0.95, 0.98), cl);
  vec3 h = normalize(u_L + vec3(0.0, 0.0, 1.0));
  float spec = pow(max(dot(n, h), 0.0), 40.0) * 0.35 * (1.0 - cl) * step(0.0, d);
  vec3 c = alb * diff * 0.86 + u_rim * (rim * 0.9 + back) + alb * (0.06 + 0.1 * n.z) + vec3(spec); // + faint starlight from the viewer's side
  float a = u_alpha * clamp((1.0 - rr) / u_aa, 0.0, 1.0);
  gl_FragColor = vec4(c * u_exp * a, a);
}`;
  // a planet's ring: a plane through the planet's centre (normal u_N, view space), drawn in two halves around the
  // sphere (behind it, then in front), with gaps, and the planet's shadow falling across it
  const RING_VS = `
attribute vec2 a_q; uniform vec2 u_c; uniform vec2 u_r;
varying vec2 v_q;
void main() { v_q = a_q * 2.5; gl_Position = vec4(u_c + a_q * u_r * 2.5, 0.0, 1.0); }`;
  const RING_FS = `
precision highp float;
varying vec2 v_q; uniform vec3 u_N; uniform vec3 u_L; uniform vec3 u_col; uniform float u_side; uniform float u_exp; uniform float u_alpha; uniform float u_seed;
` + NOISE + `
void main() {
  vec2 q = v_q; float nz = u_N.z; if (abs(nz) < 0.03) nz = nz < 0.0 ? -0.03 : 0.03;
  vec3 P = vec3(q, -(u_N.x * q.x + u_N.y * q.y) / nz);
  if (P.z * u_side < 0.0) discard;
  float rr = length(P); if (rr < 1.38 || rr > 2.4) discard;
  float band = noise(vec2(rr * 16.0 + u_seed, 0.5)) * 0.6 + noise(vec2(rr * 61.0, 1.5)) * 0.4;
  float dens = smoothstep(1.38, 1.5, rr) * smoothstep(2.4, 2.22, rr) * (0.3 + 0.7 * band);
  dens *= 1.0 - 0.85 * smoothstep(1.78, 1.82, rr) * smoothstep(1.92, 1.88, rr); // the gap
  float b = dot(P, u_L), c = dot(P, P) - 1.0, sh = (b * b - c > 0.0 && b < 0.0) ? 0.1 : 1.0;
  float lit = 0.25 + 0.75 * abs(dot(normalize(u_N), u_L));
  float a = dens * 0.8 * u_alpha;
  gl_FragColor = vec4(u_col * dens * lit * sh * u_exp * u_alpha, a);
}`;
  // lines: the warp's streaks (a star stretched along the travel, fading toward its far end) and the orbit rings
  const LINE_VS = `
attribute vec3 a_pos; attribute vec3 a_col; attribute float a_end;
uniform mat4 u_mvp; uniform vec3 u_dir; uniform float u_len; uniform vec2 u_anchor; uniform float u_exp; uniform float u_far;
varying vec3 v_col; varying float v_a;
void main() {
  vec4 c = u_mvp * vec4(a_pos + u_dir * u_len * a_end, 1.0);
  c.xy += u_anchor * c.w; gl_Position = c;
  v_a = (1.0 - a_end * 0.92) * u_exp * clamp(u_far / max(c.w, 1.0), 0.0, 1.0);
  v_col = a_col;
}`;
  const LINE_FS = `
precision mediump float;
varying vec3 v_col; varying float v_a;
void main() { gl_FragColor = vec4(v_col * v_a, v_a); }`;

  // ringed planets (index into the six): the ring plane's normal in galaxy space
  const RINGN = { 1: [0.22, 0.95, 0.2], 4: [-0.3, 0.92, 0.25] };
  function rng(seed) { let s = seed >>> 0 || 1; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
  function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.2831853 * v); }

  // mat4 helpers (column-major, as WebGL wants)
  const persp = (fov, asp, n, f) => { const t = 1 / Math.tan(fov / 2), nf = 1 / (n - f); return [t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) * nf, -1, 0, 0, 2 * f * n * nf, 0]; };
  const mul = (a, b) => { const o = new Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k]; o[i * 4 + j] = s; } return o; };
  const lookAt = (e, t, up) => {
    let zx = e[0] - t[0], zy = e[1] - t[1], zz = e[2] - t[2]; let l = Math.hypot(zx, zy, zz) || 1; zx /= l; zy /= l; zz /= l;
    let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx; l = Math.hypot(xx, xy, xz) || 1; xx /= l; xy /= l; xz /= l;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    return [xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0, -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1];
  };
  const rotY = (a) => { const c = Math.cos(a), s = Math.sin(a); return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]; };

  class VegaGalaxy extends HTMLElement {
    connectedCallback() {
      if (this.cv) return;
      if (getComputedStyle(this).position === 'static') this.style.position = 'absolute';
      this.style.display = 'block'; this.style.inset = this.style.inset || '0'; this.style.pointerEvents = 'none';
      this.setAttribute('aria-hidden', 'true');
      const cv = this.cv = document.createElement('canvas'); cv.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block';
      this.appendChild(cv);
      const gl = this.gl = cv.getContext('webgl', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false });
      if (!gl || !this.build()) { this.style.display = 'none'; this.failed = true; return; } // no WebGL: the canvas 2D sky stays
      this.view = { dist: 2600, yaw: 0.4, pitch: 0.55, ax: 0, ay: 0 }; this.exposure = 1; this.t0 = performance.now(); this.last = 0;
      this.loop = this.loop.bind(this); this.onVis = () => { if (!document.hidden) this.kick(); };
      document.addEventListener('visibilitychange', this.onVis);
      this.kick();
    }
    disconnectedCallback() { cancelAnimationFrame(this.raf); this.raf = 0; document.removeEventListener('visibilitychange', this.onVis); }
    kick() { if (!this.raf && !document.hidden && !this.failed) this.raf = requestAnimationFrame(this.loop); }
    color(name, f) { const v = getComputedStyle(this).getPropertyValue(name).trim(); if (v[0] === '#') { const n = parseInt(v.slice(1, 7), 16); return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255]; } return f; }
    prog(vs, fs) {
      const gl = this.gl, sh = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s); return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null; };
      const v = sh(gl.VERTEX_SHADER, vs), f = sh(gl.FRAGMENT_SHADER, fs); if (!v || !f) return null;
      const p = gl.createProgram(); gl.attachShader(p, v); gl.attachShader(p, f); gl.linkProgram(p); return gl.getProgramParameter(p, gl.LINK_STATUS) ? p : null;
    }
    // the galaxy: four arms on a log spiral (the mark's four spikes, echoed), a bulge and a thin halo; seeded, so every
    // page shows the same sky
    build() {
      const gl = this.gl; this.ps = this.prog(STAR_VS, STAR_FS); this.pc = this.prog(CORE_VS, CORE_FS); if (!this.ps || !this.pc) return false;
      const FG = this.color('--fg', [0.91, 0.94, 0.97]), ACC = this.color('--acc', [0.61, 0.78, 1]), CY = this.color('--cy', [0.44, 0.88, 0.86]), DIM = this.color('--dim', [0.55, 0.62, 0.7]);
      this.FG = FG; this.ACC = ACC;
      const R = 1000, ARMS = 4, r = rng(4242), pos = [], col = [], size = [], seed = [], alpha = [];
      const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
      const push = (x, y, z, c, s, al) => { pos.push(x, y, z); col.push(c[0], c[1], c[2]); size.push(s); seed.push(r()); alpha.push(al == null ? 1 : al); };
      for (let i = 0; i < 26000; i++) {
        const rr = R * (0.06 + 0.94 * Math.pow(r(), 0.72)), k = i % ARMS, spread = 0.38 * (1 - 0.45 * rr / R);
        const th = k * 6.2831853 / ARMS + Math.log(rr / 50) * 2.1 + gauss(r) * spread;
        const y = gauss(r) * 16 * (1 + 1.6 * (1 - rr / R)), t = rr / R, pick = r();
        const c = pick < 0.07 ? CY : pick < 0.36 ? DIM : mix(FG, ACC, clamp(t * 1.4, 0, 1));
        push(Math.cos(th) * rr, y, Math.sin(th) * rr, c, pick > 0.995 ? 3.2 : 1.0 + r() * 1.5);
      }
      for (let i = 0; i < 6000; i++) { const s = R * 0.11; push(gauss(r) * s, gauss(r) * s * 0.55, gauss(r) * s, mix(FG, ACC, r() * 0.4), 0.9 + r() * 1.3); }
      for (let i = 0; i < 3000; i++) { const u = r() * 2 - 1, a = r() * 6.2831853, rad = R * 1.5 * Math.cbrt(r()), q = Math.sqrt(1 - u * u); push(Math.cos(a) * q * rad, u * rad * 0.6, Math.sin(a) * q * rad, DIM, 0.8 + r()); }
      // the glow: large faint soft points along the arms and over the bulge — from outside they draw the spiral, from
      // inside they are the nebula the stars sit in (the stars alone vanish below a pixel at a distance)
      for (let i = 0; i < 1400; i++) {
        const rr = R * (0.08 + 0.92 * Math.pow(r(), 0.85)), k = i % ARMS, th = k * 6.2831853 / ARMS + Math.log(rr / 50) * 2.1 + gauss(r) * 0.22;
        const pick = r(), c = pick < 0.18 ? CY : mix(ACC, FG, r() * 0.3);
        push(Math.cos(th) * rr, gauss(r) * 10, Math.sin(th) * rr, c, 36 + r() * 52, 0.07 * (1 - 0.5 * rr / R));
      }
      for (let i = 0; i < 260; i++) { const s = R * 0.09; push(gauss(r) * s, gauss(r) * s * 0.5, gauss(r) * s, mix(FG, ACC, 0.3), 50 + r() * 70, 0.08); }
      // star-forming knots on the arms' outer edges: a cyan / accent haze with young stars in it
      for (let kk = 0; kk < 120; kk++) {
        const rr = R * (0.15 + 0.8 * Math.pow(r(), 0.8)), k = kk % ARMS, th = k * 6.2831853 / ARMS + Math.log(rr / 50) * 2.1 + 0.08 + gauss(r) * 0.09;
        const cx = Math.cos(th) * rr, cz = Math.sin(th) * rr, hue = r() < 0.55 ? CY : ACC, big = r(), sp = 4 + big * 9, nh = big < 0.35 ? 0 : 4 + Math.floor(big * 10);
        for (let j = 0; j < nh; j++) push(cx + gauss(r) * sp, gauss(r) * 2.5, cz + gauss(r) * sp, mix(hue, FG, r() * 0.3), 20 + r() * 16 * big, 0.045 + 0.04 * big);
        for (let j = 0; j < 3 + Math.floor(big * 6); j++) push(cx + gauss(r) * sp * 0.7, gauss(r) * 2, cz + gauss(r) * sp * 0.7, mix(FG, hue, 0.4), 1.3 + r() * 1.3);
      }
      this.n = size.length;
      // dust: dark lanes on the arms' leading edges — drawn last with an absorbing blend (black, alpha = density)
      for (let i = 0; i < 2600; i++) {
        const rr = R * (0.1 + 0.82 * Math.pow(r(), 0.9)), k = i % ARMS, th = k * 6.2831853 / ARMS + Math.log(rr / 50) * 2.1 - 0.17 + gauss(r) * 0.06;
        push(Math.cos(th) * rr, gauss(r) * 5, Math.sin(th) * rr, [0, 0, 0], 24 + r() * 38, 0.2 * (1 - 0.45 * rr / R));
      }
      this.nd = size.length - this.n;
      const buf = (data, loc, n) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW); return { b, loc, n }; };
      this.attrs = [buf(pos, gl.getAttribLocation(this.ps, 'a_pos'), 3), buf(col, gl.getAttribLocation(this.ps, 'a_col'), 3), buf(size, gl.getAttribLocation(this.ps, 'a_size'), 1), buf(seed, gl.getAttribLocation(this.ps, 'a_seed'), 1), buf(alpha, gl.getAttribLocation(this.ps, 'a_alpha'), 1)];
      this.quad = buf([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], gl.getAttribLocation(this.pc, 'a_q'), 2);
      // the warp: every point star twice, its far end stretched along the travel in the shader
      this.pb = this.prog(BODY_VS, BODY_FS); this.pl = this.prog(LINE_VS, LINE_FS); if (!this.pb || !this.pl) return false;
      this.pr = this.prog(RING_VS, RING_FS);
      this.R = {}; if (this.pr) ['u_c', 'u_r', 'u_N', 'u_L', 'u_col', 'u_side', 'u_exp', 'u_alpha', 'u_seed'].forEach(k => this.R[k] = gl.getUniformLocation(this.pr, k));
      this.pp = { bright: this.prog(FS_VS, BRIGHT_FS), blur: this.prog(FS_VS, BLUR_FS), comp: this.prog(FS_VS, COMP_FS) };
      this.post = !!(this.pp.bright && this.pp.blur && this.pp.comp);
      const NS = 35000, lp = [], lc = [], le = [];
      for (let i = 0; i < NS; i++) for (let e = 0; e < 2; e++) { lp.push(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]); lc.push(col[i * 3], col[i * 3 + 1], col[i * 3 + 2]); le.push(e); }
      this.nl = NS * 2;
      this.lattrs = [buf(lp, gl.getAttribLocation(this.pl, 'a_pos'), 3), buf(lc, gl.getAttribLocation(this.pl, 'a_col'), 3), buf(le, gl.getAttribLocation(this.pl, 'a_end'), 1)];
      this.L = {}; ['u_mvp', 'u_dir', 'u_len', 'u_anchor', 'u_exp', 'u_far'].forEach(k => this.L[k] = gl.getUniformLocation(this.pl, k));
      this.B = {}; ['u_c', 'u_r', 'u_L', 'u_base', 'u_band', 'u_rim', 'u_exp', 'u_seed', 'u_spin', 'u_alpha', 'u_aa', 'u_t'].forEach(k => this.B[k] = gl.getUniformLocation(this.pb, k));
      this.bq = buf([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], gl.getAttribLocation(this.pb, 'a_q'), 2);
      // the planets' looks, from VEGA's palette: [surface, bands, atmosphere]
      const sc = (c, k) => c.map(v => v * k);
      this.PAL = [[sc(DIM, 0.5), FG, ACC], [sc(ACC, 0.45), sc(FG, 0.9), ACC], [sc(CY, 0.4), DIM, CY], [sc(FG, 0.5), ACC, ACC], [sc(DIM, 0.45), CY, CY], [sc(ACC, 0.4), DIM, ACC]];
      // the VEGA system in galaxy space (flight): six planets, INDEX order, on slightly warped orbits, and their rings
      this.SYS = [16, 24, 33, 43, 54, 66].map((r, k) => ({ r, ph: k / 6 * 6.2831853 + 0.35, size: [2.0, 2.6, 2.3, 3.2, 2.5, 2.9][k], w: 0.05 / Math.sqrt(r / 16) }));
      const rp = []; this.SYS.forEach(o => { for (let j = 0; j < 160; j++) { const th = j / 160 * 6.2831853; rp.push(...this.orbitAt(o, th)); } });
      this.rings = buf(rp, gl.getAttribLocation(this.pl, 'a_pos'), 3); this.DIM = DIM;
      this.U = {}; ['u_mvp', 'u_anchor', 'u_px', 'u_time', 'u_tw', 'u_exp'].forEach(k => this.U[k] = gl.getUniformLocation(this.ps, k));
      this.C = {}; ['u_c', 'u_r', 'u_R', 'u_S', 'u_L', 'u_k', 'u_in', 'u_t', 'u_flare', 'u_fg', 'u_acc', 'u_exp', 'u_pulse'].forEach(k => this.C[k] = gl.getUniformLocation(this.pc, k));
      return true;
    }
    // render targets: the scene at full size, the bloom at a quarter and an eighth (ping-pong pairs)
    target(w, h) {
      const gl = this.gl;
      if (this.T && this.T.w === w && this.T.h === h) return this.T.ok;
      const mk = (tw, th) => { const tx = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tx);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, tw, th, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER].forEach(k => gl.texParameteri(gl.TEXTURE_2D, k, gl.LINEAR));
        [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T].forEach(k => gl.texParameteri(gl.TEXTURE_2D, k, gl.CLAMP_TO_EDGE));
        const fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tx, 0);
        return { tx, fb, w: tw, h: th }; };
      if (this.T) ['scene', 'a1', 'b1', 'a2', 'b2'].forEach(k => { gl.deleteTexture(this.T[k].tx); gl.deleteFramebuffer(this.T[k].fb); });
      const q = (n) => Math.max(1, Math.round(n));
      this.T = { w, h, scene: mk(w, h), a1: mk(q(w / 4), q(h / 4)), b1: mk(q(w / 4), q(h / 4)), a2: mk(q(w / 8), q(h / 8)), b2: mk(q(w / 8), q(h / 8)) };
      this.T.ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      if (!this.T.ok) this.post = false;
      return this.T.ok;
    }
    bloom(w, h, t, ex) {
      const gl = this.gl, T = this.T;
      gl.disable(gl.BLEND);
      const pass = (prog, dst, set) => { gl.useProgram(prog); gl.bindFramebuffer(gl.FRAMEBUFFER, dst ? dst.fb : null); gl.viewport(0, 0, dst ? dst.w : w, dst ? dst.h : h);
        const loc = gl.getAttribLocation(prog, 'a_q'); set((n) => gl.getUniformLocation(prog, n));
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quad.b); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0); gl.drawArrays(gl.TRIANGLES, 0, 6); gl.disableVertexAttribArray(loc); };
      const tex = (u, unit, t2) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t2.tx); gl.uniform1i(u, unit); };
      const { bright, blur, comp } = this.pp;
      pass(bright, T.a1, (U) => { tex(U('u_tex'), 0, T.scene); gl.uniform2f(U('u_px'), 1.5 / w, 1.5 / h); gl.uniform1f(U('u_th'), 0.5); });
      const blur2 = (a, b, k) => { pass(blur, b, (U) => { tex(U('u_tex'), 0, a); gl.uniform2f(U('u_dir'), k / a.w, 0); }); pass(blur, a, (U) => { tex(U('u_tex'), 0, b); gl.uniform2f(U('u_dir'), 0, k / b.h); }); };
      blur2(T.a1, T.b1, 1); blur2(T.a1, T.b1, 2);
      pass(bright, T.a2, (U) => { tex(U('u_tex'), 0, T.a1); gl.uniform2f(U('u_px'), 1 / T.a1.w, 1 / T.a1.h); gl.uniform1f(U('u_th'), -1); });
      blur2(T.a2, T.b2, 1.5); blur2(T.a2, T.b2, 3);
      pass(comp, null, (U) => { tex(U('u_scene'), 0, T.scene); tex(U('u_b1'), 1, T.a1); tex(U('u_b2'), 2, T.a2);
        gl.uniform2f(U('u_res'), w, h); gl.uniform1f(U('u_t'), RM.matches ? 0 : t); gl.uniform1f(U('u_bk'), this.bloomK == null ? 0.9 : this.bloomK); gl.uniform1f(U('u_grain'), RM.matches ? 0.02 : 0.045); });
      gl.activeTexture(gl.TEXTURE0); gl.enable(gl.BLEND);
    }
    orbitAt(o, th) { return [Math.cos(th) * o.r, Math.sin(th * 2 + o.ph) * o.r * 0.06, Math.sin(th) * o.r]; }
    // the planets' galaxy-space positions at time t (s); still under reduced motion
    planets(t) { if (!this.SYS) return []; const tt = RM.matches ? 0 : (t == null ? (performance.now() - this.t0) / 1000 : t);
      return this.SYS.map((o, k) => ({ k, p: this.orbitAt(o, o.ph + tt * o.w), size: o.size })); }
    loop(now) {
      this.raf = 0; if (document.hidden) { this._qt = 0; return; } // a pause (hidden tab) isn't a slow frame
      // quality governor: a machine that can't hold the frame (software GL) steps the galaxy down — no bloom, then a
      // lower resolution, then every other frame — and never back up (no oscillation). Software renderers start at 2.
      if (this.q == null) { this.q = 0; try { const g0 = this.gl, base = String(g0.getParameter(g0.RENDERER) || ''); let rn = base; /* Firefox reports the real renderer here and deprecates the extension; Chrome says 'WebKit WebGL' */
        if (/^webkit webgl$/i.test(base)) { const ext = g0.getExtension('WEBGL_debug_renderer_info'); if (ext) rn = String(g0.getParameter(ext.UNMASKED_RENDERER_WEBGL)); } if (/swiftshader|llvmpipe|software/i.test(rn)) this.q = 2; this.renderer = rn; } catch (e) {} }
      if (this._qt && now - this._qt < 250) { const fdt = now - this._qt; /* gaps (tab switch, a long task) don't count */ this._fe = this._fe == null ? fdt : this._fe + (fdt - this._fe) * 0.05; this._slow = this._fe > 26 ? (this._slow || 0) + 1 : 0;
        if (this._slow > 90 && this.q < 3) { this.q++; this._slow = 0; this._fe = 16.7; } }
      this._qt = now;
      if (this.q >= 3 && (this._skip = !this._skip)) { this.kick(); return; } // half rate: the galaxy is a slow background
      // the galaxy is soft light: it renders below the screen's resolution (el.res, default 0.6) and the page scales it up
      const gl = this.gl, cv = this.cv, rm = RM.matches, dpr = Math.min(2, devicePixelRatio || 1) * (this.res || 0.6) * (this.q >= 2 ? 0.55 : 1);
      const W = cv.clientWidth, H = cv.clientHeight; if (!W || !H) { this.kick(); return; }
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
      const dt = Math.min(0.05, this.last ? (now - this.last) / 1000 : 0.016); this.last = now;
      // ease toward the driver's view (a cut under reduced motion)
      const d = this.drive ? this.drive(W, H) : null, v = this.view;
      if (d) {
        const tgt = { dist: d.dist, yaw: d.yaw, pitch: d.pitch, ax: d.anchor ? d.anchor[0] / W * 2 - 1 : 0, ay: d.anchor ? 1 - d.anchor[1] / H * 2 : 0 };
        if (rm || !this._eased) { Object.assign(v, tgt); this._eased = true; }
        else { const k = 1 - Math.exp(-dt * 8); /* G8: was 5.5 */ v.yaw += (tgt.yaw - v.yaw) * k; v.pitch += (tgt.pitch - v.pitch) * k; v.ax += (tgt.ax - v.ax) * k; v.ay += (tgt.ay - v.ay) * k;
          v.dist = Math.exp(Math.log(v.dist) + (Math.log(tgt.dist) - Math.log(v.dist)) * k); } // distance eases in log space: the dive feels even
      }
      if (this.tick) this.tick(dt, now); // a pilot (vega-flight) moves el.cam inside this frame
      // home: an inner page is its planet's neighbourhood — the camera floats beside the page's planet in the VEGA system,
      // VEGA and the galaxy in the sky, and orbits a little as the page scrolls (el.homeU() 0..1); scroll shots streak
      if (this.home != null && !this.cam && this.SYS) {
        const pl = this.planets()[this.home], p = pl.p, sz = pl.size, u = this.homeU ? clamp(this.homeU(), 0, 1) : 0;
        const o = [p[0], 0, p[2]], ol = Math.hypot(o[0], o[2]) || 1; o[0] /= ol; o[2] /= ol;
        const a = 0.2 + u * 0.7, ca = Math.cos(a), sa = Math.sin(a), sd = [o[2], 0, -o[0]];
        const dir = [sd[0] * ca - o[0] * sa, 0, sd[2] * ca - o[2] * sa]; // around the planet's sunward flank: the planet stays behind the lens, never across VEGA
        const eye = [p[0] + dir[0] * sz * 8, p[1] + sz * 1.4, p[2] + dir[2] * sz * 8];
        // the camera looks so that VEGA sits pinned in the stage's empty upper-left (above the rail, below the brand):
        // the orbit moves the galaxy band and the other bodies behind it, never the bright star into the text or the scene
        const port = W / H < 0.8, tfh = Math.tan(0.45), X = (this.homeAt ? this.homeAt[0] : port ? 0.5 : -0.62) * tfh * (W / H), Y = (this.homeAt ? this.homeAt[1] : port ? 0.62 : 0.6) * tfh; // a phone's top-left holds the page's HUD block: VEGA goes upper right
        const dv0 = [-eye[0], -eye[1], -eye[2]], dl = Math.hypot(dv0[0], dv0[1], dv0[2]) || 1, dv = [dv0[0] / dl, dv0[1] / dl, dv0[2] / dl], kq = Math.sqrt(1 + X * X + Y * Y);
        let f = dv.slice();
        for (let it = 0; it < 4; it++) { let rx = f[1] * 0 - f[2] * 1, ry = f[2] * 0 - f[0] * 0, rz = f[0] * 1 - f[1] * 0; const rl = Math.hypot(rx, ry, rz) || 1; rx /= rl; ry /= rl; rz /= rl; // right = fwd × up
          const ux = ry * f[2] - rz * f[1], uy = rz * f[0] - rx * f[2], uz = rx * f[1] - ry * f[0];
          f = [dv[0] * kq - rx * X - ux * Y, dv[1] * kq - ry * X - uy * Y, dv[2] * kq - rz * X - uz * Y]; const fl = Math.hypot(f[0], f[1], f[2]); f = [f[0] / fl, f[1] / fl, f[2] / fl]; }
        const tg = [eye[0] + f[0], eye[1] + f[1], eye[2] + f[2]];
        // arriving through the warp (vega-fly), the camera flies in to the page's view from further out instead of starting there
        if (!this._hc && !rm && window.VegaFly && window.VegaFly.arrived) { const bo = [eye[0] - p[0], 0, eye[2] - p[2]], bl = Math.hypot(bo[0], bo[2]) || 1; this._hc = { eye: [p[0] + bo[0] / bl * sz * 46, p[1] + sz * 14, p[2] + bo[2] / bl * sz * 46], tg: p.slice() }; this._arr = now; }
        const hold = this._arr && window.VegaFly && window.VegaFly.arriving; if (hold) this._arr = now; // held out there until the cover starts to lift
        const hc = this._hc || (this._hc = { eye: eye.slice(), tg: tg.slice() }), kk = rm ? 1 : hold ? 0 : 1 - Math.exp(-dt * (this._arr && now - this._arr < 3200 ? 1.15 : 3));
        for (let j = 0; j < 3; j++) { hc.eye[j] += (eye[j] - hc.eye[j]) * kk; hc.tg[j] += (tg[j] - hc.tg[j]) * kk; }
        this.homeCam = { eye: hc.eye, fwd: [hc.tg[0] - hc.eye[0], hc.tg[1] - hc.eye[1], hc.tg[2] - hc.eye[2]], up: [0, 1, 0] };
        const R = window.VegaReel; this.streak = R && R.playing ? Math.sin(Math.PI * (R.t || 0)) * 0.3 : 0;
      } else this.homeCam = null;
      const t = (now - this.t0) / 1000, spin = rm ? 0 : t * 0.006;
      const nrm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
      let eye, fwd, up = [0, 1, 0], ax = v.ax, ay = v.ay;
      const cm = this.cam || this.homeCam;
      if (cm) { eye = cm.eye.slice(); fwd = nrm(cm.fwd); up = cm.up || up; ax = cm.ax || 0; ay = cm.ay || 0; } // (a flight that leaves a page eases the page's anchor out)
      else { eye = [Math.sin(v.yaw) * Math.cos(v.pitch) * v.dist, Math.sin(v.pitch) * v.dist, Math.cos(v.yaw) * Math.cos(v.pitch) * v.dist]; fwd = nrm(eye.map(c => -c)); }
      // the warp: the camera runs forward along its view, faster with the square of el.warp
      const wp = rm ? 0 : clamp(this.warp || 0, 0, 1.2), R0 = Math.hypot(eye[0], eye[1], eye[2]);
      if (wp > 0) this.wo = (this.wo || 0) + wp * wp * Math.max(600, R0 * 6) * dt; else this.wo = 0;
      if (this.wo) eye = eye.map((c, k) => c + fwd[k] * this.wo);
      if (this.warpAt && wp > 0) { const k = clamp(wp * 2, 0, 1), k2 = k * k * (3 - 2 * k); ax += (this.warpAt[0] / W * 2 - 1 - ax) * k2; ay += (1 - this.warpAt[1] / H * 2 - ay) * k2; }
      const near = cm ? 0.05 : Math.max(1, v.dist * 0.02), FOV = 0.9;
      const P = persp(FOV, W / H, near, 20000), V = lookAt(eye, eye.map((c, k) => c + fwd[k]), up), PV = mul(P, V), MVP = mul(PV, rotY(spin));
      const post = this.post && !this.noPost && this.q < 1 && this.target(cv.width, cv.height); // the scene goes to a texture for the bloom
      gl.bindFramebuffer(gl.FRAMEBUFFER, post ? this.T.scene.fb : null);
      gl.viewport(0, 0, cv.width, cv.height); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); // additive light on a transparent canvas (premultiplied)
      const ex = clamp(this.exposure == null ? 1 : this.exposure, 0, 1);
      gl.useProgram(this.ps);
      gl.uniformMatrix4fv(this.U.u_mvp, false, new Float32Array(MVP)); gl.uniform2f(this.U.u_anchor, ax, ay); gl.uniform1f(this.U.u_px, dpr);
      gl.uniform1f(this.U.u_time, t); gl.uniform1f(this.U.u_tw, rm ? 0 : 1); gl.uniform1f(this.U.u_exp, ex * 0.85 * (1 - 0.55 * clamp(wp, 0, 1)) * (this.field == null ? 1 : clamp(this.field, 0, 1))); // el.field: the star field alone (not VEGA, its mark or the planets) — index's hero keeps it quiet
      this.attrs.forEach(a => { gl.bindBuffer(gl.ARRAY_BUFFER, a.b); gl.enableVertexAttribArray(a.loc); gl.vertexAttribPointer(a.loc, a.n, gl.FLOAT, false, 0, 0); });
      gl.drawArrays(gl.POINTS, 0, this.n);
      gl.blendFunc(gl.ZERO, gl.ONE_MINUS_SRC_ALPHA); gl.drawArrays(gl.POINTS, this.n, this.nd); gl.blendFunc(gl.ONE, gl.ONE); // dust absorbs
      this.attrs.forEach(a => gl.disableVertexAttribArray(a.loc));
      const lines = (mvp, len, dir, exp, far) => { gl.useProgram(this.pl); gl.uniformMatrix4fv(this.L.u_mvp, false, new Float32Array(mvp)); gl.uniform3fv(this.L.u_dir, dir); gl.uniform1f(this.L.u_len, len); gl.uniform2f(this.L.u_anchor, ax, ay); gl.uniform1f(this.L.u_exp, exp); gl.uniform1f(this.L.u_far, far); };
      const sk = rm ? 0 : clamp(this.streak || 0, 0, 1); // speed lines in a scroll shot (vega-reel), with no travel of their own
      if (wp > 0.01 || sk > 0.01) { // streaks: the far end runs ahead along the travel (in the galaxy's own turning frame)
        const c = Math.cos(spin), sn = Math.sin(spin), dm = [c * fwd[0] - sn * fwd[2], fwd[1], sn * fwd[0] + c * fwd[2]];
        if (wp > 0.01) lines(MVP, wp * wp * Math.max(160, R0 * 1.6), dm, ex * (this.post ? 0.2 : 0.32) * clamp(wp * 1.6, 0, 1) * (1 - 0.35 * clamp(wp - 0.6, 0, 1)), Math.max(2600, R0 * 12));
        else lines(MVP, sk * R0 * 0.06, dm, ex * 0.11 * sk, Math.max(2600, R0 * 12));
        this.lattrs.forEach(a => { gl.bindBuffer(gl.ARRAY_BUFFER, a.b); gl.enableVertexAttribArray(a.loc); gl.vertexAttribPointer(a.loc, a.n, gl.FLOAT, false, 0, 0); });
        gl.drawArrays(gl.LINES, 0, this.nl);
        this.lattrs.forEach(a => gl.disableVertexAttribArray(a.loc));
      }
      // the VEGA system (flight): orbit rings, then the planets as bodies
      let bodies = null, coreZ = 0;
      const vw = (p) => [V[0] * p[0] + V[4] * p[1] + V[8] * p[2] + V[12], V[1] * p[0] + V[5] * p[1] + V[9] * p[2] + V[13], V[2] * p[0] + V[6] * p[1] + V[10] * p[2] + V[14]];
      const vc = vw([0, 0, 0]), tf = Math.tan(FOV / 2);
      if (this.system) {
        lines(PV, 0, [0, 0, 0], ex * 0.32, 400);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.rings.b); gl.enableVertexAttribArray(this.rings.loc); gl.vertexAttribPointer(this.rings.loc, 3, gl.FLOAT, false, 0, 0);
        const lc = gl.getAttribLocation(this.pl, 'a_col'); gl.vertexAttrib3f(lc, this.DIM[0], this.DIM[1], this.DIM[2]);
        for (let k = 0; k < 6; k++) gl.drawArrays(gl.LINE_LOOP, k * 160, 160);
        gl.disableVertexAttribArray(this.rings.loc);
        coreZ = -vc[2];
        bodies = this.planets(t).map(b => { const pv = vw(b.p), z = -pv[2]; if (z < near * 2) return null;
          const L = nrm([vc[0] - pv[0], vc[1] - pv[1], vc[2] - pv[2]]);
          const wn = RINGN[b.k], N = wn ? [V[0] * wn[0] + V[4] * wn[1] + V[8] * wn[2], V[1] * wn[0] + V[5] * wn[1] + V[9] * wn[2], V[2] * wn[0] + V[6] * wn[1] + V[10] * wn[2]] : null;
          const nx = pv[0] / (z * tf * W / H) + ax, ny = pv[1] / (z * tf) + ay, ry = b.size / (z * tf);
          // an inner page's sky keeps VEGA clear: a body crossing in front of the pinned star fades (flight keeps real eclipses)
          let al = 1; if (this.homeCam && -vc[2] > z) { const cxn = vc[0] / (-vc[2] * tf * W / H) + ax, cyn = vc[1] / (-vc[2] * tf) + ay, dd = Math.hypot((nx - cxn) * W / H, ny - cyn); al = clamp((dd / ry - 1.3) / 1.2, 0, 1); }
          if (this.homeCam) al *= clamp((0.2 - ry) / 0.1, 0, 1); // and a neighbour that would fill the frame stays out of the page's way
          if (this.homeCam) { const nav = 1 - 2 * ((W <= 600 ? 12 : 24) + 64) / H; al *= clamp((nav - (ny + ry * 0.6)) / Math.max(0.02, ry), 0, 1); } // nor sits behind the nav row and the brand (G7: no band there)
          return { nx, ny, ry, L, z, i: b.k, a: al, N }; }).filter(Boolean);
      } else if (this.bodies) {
        const list = this.bodies(W, H); if (list) { coreZ = list.coreZ || 0;
          bodies = list.map(b => ({ nx: b.x / W * 2 - 1, ny: 1 - b.y / H * 2, ry: b.r / H * 2, L: nrm(b.L), z: b.z, i: b.i, a: b.a == null ? 1 : b.a, N: RINGN[b.i] ? nrm([0.18, 0.93, 0.32]) : null })); }
      }
      // depth order: bodies beyond VEGA, the star, the bodies in front of it
      const body = (b) => {
        // in a jump the bodies rush at the lens and are gone before the flash
        if (wp > 0) { b = Object.assign({}, b, { ry: b.ry * (1 + 6 * wp * wp), a: b.a * (1 - clamp((wp - 0.55) / 0.45, 0, 1)) }); }
        if (b.a < 0.01 || b.ry * H * dpr < 0.6) return;
        const pal = this.PAL[b.i % this.PAL.length], ry = b.ry, rx = ry * H / W;
        gl.useProgram(this.pb); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.uniform2f(this.B.u_c, b.nx, b.ny); gl.uniform2f(this.B.u_r, rx, ry); gl.uniform3fv(this.B.u_L, b.L);
        gl.uniform3fv(this.B.u_base, pal[0]); gl.uniform3fv(this.B.u_band, pal[1]); gl.uniform3fv(this.B.u_rim, pal[2]);
        gl.uniform1f(this.B.u_exp, ex); gl.uniform1f(this.B.u_seed, b.i * 7.31); gl.uniform1f(this.B.u_spin, rm ? 0 : t * 0.05 * (1 + b.i * 0.13));
        gl.uniform1f(this.B.u_alpha, clamp(b.a, 0, 1)); gl.uniform1f(this.B.u_aa, Math.max(0.02, 1.5 / (ry * H * dpr / 2)));
        gl.bindBuffer(gl.ARRAY_BUFFER, this.bq.b); gl.enableVertexAttribArray(this.bq.loc); gl.vertexAttribPointer(this.bq.loc, 2, gl.FLOAT, false, 0, 0);
        gl.uniform1f(this.B.u_t, rm ? 0 : t);
        ring(b, -1, rx, ry);
        gl.useProgram(this.pb);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.bq.b); gl.enableVertexAttribArray(this.bq.loc); gl.vertexAttribPointer(this.bq.loc, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 6); gl.disableVertexAttribArray(this.bq.loc);
        ring(b, 1, rx, ry);
      };
      const ring = (b, side, rx, ry) => { if (!b.N || !this.pr) return;
        gl.useProgram(this.pr); const rl = gl.getAttribLocation(this.pr, 'a_q');
        gl.uniform2f(this.R.u_c, b.nx, b.ny); gl.uniform2f(this.R.u_r, rx, ry); gl.uniform3fv(this.R.u_N, nrm(b.N)); gl.uniform3fv(this.R.u_L, b.L);
        gl.uniform3fv(this.R.u_col, this.PAL[b.i % this.PAL.length][1]); gl.uniform1f(this.R.u_side, side); gl.uniform1f(this.R.u_exp, ex); gl.uniform1f(this.R.u_alpha, clamp(b.a, 0, 1)); gl.uniform1f(this.R.u_seed, b.i * 3.7);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.bq.b); gl.enableVertexAttribArray(rl); gl.vertexAttribPointer(rl, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 6); gl.disableVertexAttribArray(rl); };
      const sorted = bodies ? bodies.slice().sort((a, b) => b.z - a.z) : [];
      sorted.filter(b => b.z > coreZ).forEach(body);
      // the VEGA star, sized by distance (a point far off, a sun up close), at the projected core
      const cz = -vc[2];
      if (cz > near) {
        const cx = vc[0] / (cz * tf * W / H) + ax, cy = vc[1] / (cz * tf) + ay, dc = Math.hypot(eye[0], eye[1], eye[2]);
        const S = clamp(5200 / Math.max(dc, 1), 14, 420) * dpr, mk = this.mark || null, L = mk && mk.k > 0.001 ? mk.L * dpr : 0, fl = rm ? 0 : clamp(this.flare == null ? 0.35 : this.flare, 0, 1);
        const Rx = Math.max(S, L * 1.4, fl > 0.01 ? S * 9 : 0), Ry = Math.max(S, L * 1.4);
        const pk = this.pulseAt ? Math.exp(-(now - this.pulseAt) / 450) : 0;
        gl.useProgram(this.pc); gl.blendFunc(gl.ONE, gl.ONE);
        gl.uniform2f(this.C.u_c, cx, cy); gl.uniform2f(this.C.u_r, Rx / cv.width * 2, Ry / cv.height * 2); gl.uniform2f(this.C.u_R, Rx, Ry);
        gl.uniform1f(this.C.u_S, S); gl.uniform1f(this.C.u_L, L); gl.uniform1f(this.C.u_k, mk ? clamp(mk.k, 0, 1) : 0); gl.uniform1f(this.C.u_in, mk ? clamp(mk.in == null ? 1 : mk.in, 0, 1) : 1);
        gl.uniform1f(this.C.u_t, rm ? 0 : t); gl.uniform1f(this.C.u_flare, fl);
        gl.uniform3fv(this.C.u_fg, this.FG); gl.uniform3fv(this.C.u_acc, this.ACC); gl.uniform1f(this.C.u_exp, ex); gl.uniform1f(this.C.u_pulse, rm ? 0.5 : clamp(0.5 + 0.5 * Math.sin(t * 1.3) * 0.6 + pk, 0, 2));
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quad.b); gl.enableVertexAttribArray(this.quad.loc); gl.vertexAttribPointer(this.quad.loc, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 6); gl.disableVertexAttribArray(this.quad.loc);
        this.star = { x: (cx + 1) / 2 * cv.width / dpr, y: (1 - cy) / 2 * cv.height / dpr, r: S / dpr }; // in css px of the canvas, for text lit by VEGA (vega-reel)
      } else this.star = null;
      sorted.filter(b => b.z <= coreZ).forEach(body);
      if (post) this.bloom(cv.width, cv.height, t, ex);
      this.frame = { eye, fwd, PV, ax, ay, W, H }; // for a page that projects into this view (flight labels)
      if (this.onframe) this.onframe(this.frame); // (not `after`: that is Element.after(), which inserts text siblings)
      this.kick();
    }
  }
  customElements.define('vega-galaxy', VegaGalaxy);
})();
