// dsp.js - BirdCLEF front end in plain JavaScript (works in the browser and in Node).
// Mirrors: librosa.feature.melspectrogram(power=2) -> librosa.power_to_db(ref=np.max, top_db) -> (x + 40) / 40
(function (root) {
  'use strict';

  const plans = {};
  function getPlan(n) {
    if (plans[n]) return plans[n];
    const levels = Math.round(Math.log2(n));
    if ((1 << levels) !== n) throw new Error('FFT size must be a power of 2');
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0, x = i;
      for (let j = 0; j < levels; j++) { r = (r << 1) | (x & 1); x >>= 1; }
      rev[i] = r;
    }
    return (plans[n] = { n, cos, sin, rev });
  }

  function fft(plan, re, im) {                       // in-place forward complex FFT
    const { n, cos, sin, rev } = plan;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const l = j + half;
          const tre = re[l] * cos[k] + im[l] * sin[k];
          const tim = im[l] * cos[k] - re[l] * sin[k];
          re[l] = re[j] - tre; im[l] = im[j] - tim;
          re[j] += tre; im[j] += tim;
        }
      }
    }
  }

  function padSignal(y, pad, mode) {                 // librosa center=True padding
    const out = new Float32Array(y.length + 2 * pad);
    out.set(y, pad);
    if (mode === 'reflect') {
      for (let i = 0; i < pad; i++) {
        out[pad - 1 - i] = y[Math.min(1 + i, y.length - 1)];
        out[pad + y.length + i] = y[Math.max(y.length - 2 - i, 0)];
      }
    }
    return out;                                      // 'constant' -> zeros
  }

  // y: Float32Array at cfg.sr. fb: Float32Array (nMels * (nFft/2+1)), win: Float32Array (nFft)
  // returns Float32Array (nMels * nFrames), row-major [mel][frame], in dB
  function melDb(y, cfg, fb, win) {
    const nFft = cfg.nFft, hop = cfg.hop, nMels = cfg.nMels, nBins = nFft / 2 + 1;
    const nFrames = 1 + Math.floor(y.length / hop);
    const yp = padSignal(y, nFft / 2, cfg.padMode);
    const plan = getPlan(nFft);
    const lo = new Int32Array(nMels), hi = new Int32Array(nMels);
    for (let m = 0; m < nMels; m++) {                // filters are sparse triangles: skip the zeros
      let a = -1, b = -1;
      for (let k = 0; k < nBins; k++) if (fb[m * nBins + k] !== 0) { if (a < 0) a = k; b = k; }
      lo[m] = a < 0 ? 0 : a; hi[m] = b < 0 ? -1 : b;
    }
    const re = new Float64Array(nFft), im = new Float64Array(nFft), S = new Float64Array(nBins);
    const mel = new Float64Array(nMels * nFrames);
    for (let t = 0; t < nFrames; t++) {
      const off = t * hop;
      for (let i = 0; i < nFft; i++) { re[i] = yp[off + i] * win[i]; im[i] = 0; }
      fft(plan, re, im);
      for (let k = 0; k < nBins; k++) S[k] = re[k] * re[k] + im[k] * im[k];
      for (let m = 0; m < nMels; m++) {
        let acc = 0; const base = m * nBins;
        for (let k = lo[m]; k <= hi[m]; k++) acc += fb[base + k] * S[k];
        mel[m * nFrames + t] = acc;
      }
    }
    let mx = 0;
    for (let i = 0; i < mel.length; i++) if (mel[i] > mx) mx = mel[i];
    const amin = 1e-10, refDb = 10 * Math.log10(Math.max(amin, mx));
    const db = new Float32Array(mel.length);
    let dmax = -Infinity;
    for (let i = 0; i < mel.length; i++) {
      const v = 10 * Math.log10(Math.max(amin, mel[i])) - refDb;
      db[i] = v; if (v > dmax) dmax = v;
    }
    const floor = dmax - cfg.topDb;
    for (let i = 0; i < db.length; i++) if (db[i] < floor) db[i] = floor;
    return db;
  }

  function modelInput(db) {                          // same scaling as training: (x + 40) / 40
    const out = new Float32Array(db.length);
    for (let i = 0; i < db.length; i++) out[i] = (db[i] + 40) / 40;
    return out;
  }

  // Windowed-sinc resampler (Hann window, 24 zero crossings) with an anti-alias low-pass at the output Nyquist.
  function resample(x, srIn, srOut) {
    if (srIn === srOut) return Float32Array.from(x);
    const ratio = srIn / srOut;
    const cut = srOut < srIn ? srOut / srIn : 1.0;   // cutoff relative to the input Nyquist
    const Z = 24, hw = Z / cut, R = 512;
    const n = Math.ceil(2 * hw * R) + 1, table = new Float64Array(n + 1);
    for (let i = 0; i <= n; i++) {
      const d = i / R - hw, u = d / hw;
      const s = d === 0 ? 1 : Math.sin(Math.PI * cut * d) / (Math.PI * cut * d);
      table[i] = (Math.abs(u) < 1 ? 0.5 * (1 + Math.cos(Math.PI * u)) : 0) * cut * s;
    }
    const nOut = Math.floor(x.length / ratio), out = new Float32Array(nOut);
    for (let o = 0; o < nOut; o++) {
      const p = o * ratio, k0 = Math.ceil(p - hw), k1 = Math.floor(p + hw);
      let acc = 0, wsum = 0;
      for (let k = k0; k <= k1; k++) {
        let pos = (k - p + hw) * R;
        if (pos < 0) pos = 0; else if (pos > n - 1) pos = n - 1;   // guard against rounding at the kernel edges
        const i = Math.floor(pos), f = pos - i;
        const w = table[i] * (1 - f) + table[i + 1] * f;
        wsum += w;
        if (k >= 0 && k < x.length) acc += x[k] * w;
      }
      out[o] = acc / wsum;
    }
    return out;
  }

  // full 5 s windows, plus a tail of >= 1 s repeated to fill a window (like training)
  function splitWindows(x, nSamples, sr) {
    const wins = [], full = Math.floor(x.length / nSamples);
    for (let i = 0; i < full; i++) wins.push({ start: i * nSamples / sr, data: x.subarray(i * nSamples, (i + 1) * nSamples) });
    const rem = x.subarray(full * nSamples);
    if (rem.length >= sr) {
      const w = new Float32Array(nSamples);
      for (let i = 0; i < nSamples; i++) w[i] = rem[i % rem.length];
      wins.push({ start: full * nSamples / sr, data: w });
    }
    return wins;
  }

  const api = { melDb, modelInput, resample, splitWindows, fft, getPlan };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.BirdDSP = api;
})(typeof self !== 'undefined' ? self : this);
