// app.js - on-device bird sound detector (runs entirely in the browser; audio never leaves the phone)
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  const S = { cfg: null, fb: null, win: null, classes: null, names: null, session: null, busy: false };

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function setStatus(msg, kind) { const s = $('status'); s.textContent = msg; s.className = kind || ''; }
  function setBusy(b) { S.busy = b; ['btnRec', 'btnFile', 'btnTest'].forEach(id => { $(id).disabled = b || !S.session; }); }
  const sigmoid = z => 1 / (1 + Math.exp(-z));
  const tick = () => new Promise(r => setTimeout(r, 0));

  async function fetchJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(u + ': HTTP ' + r.status); return r.json(); }
  async function fetchF32(u) { const r = await fetch(u); if (!r.ok) throw new Error(u + ': HTTP ' + r.status); return new Float32Array(await r.arrayBuffer()); }

  async function init() {
    try {
      setStatus('Loading the model (about 7 MB)…');
      if (typeof ort === 'undefined') throw new Error('The ONNX Runtime library could not be loaded (no internet?).');
      S.cfg = await fetchJSON('data/config.json');
      const lists = await Promise.all([fetchJSON('data/classes.json'), fetchJSON('data/names.json')]);
      S.classes = lists[0]; S.names = lists[1];
      const arrs = await Promise.all([fetchF32('data/mel_fb.bin'), fetchF32('data/hann.bin')]);
      S.fb = arrs[0]; S.win = arrs[1];
      ort.env.wasm.wasmPaths = ORT_BASE;
      ort.env.wasm.numThreads = 1;
      S.session = await ort.InferenceSession.create('model/student_fp32.onnx', { executionProviders: ['wasm'] });
      setBusy(false);
      setStatus('Ready. Record a few seconds of birdsong, or pick an audio file.', 'ok');
    } catch (e) {
      setStatus('Could not start: ' + e.message, 'err');
    }
  }

  async function scoreWindow(data) {
    const t0 = performance.now();
    const db = BirdDSP.melDb(data, S.cfg, S.fb, S.win);
    const t1 = performance.now();
    const frames = db.length / S.cfg.nMels;
    const feeds = { mel: new ort.Tensor('float32', BirdDSP.modelInput(db), [1, 1, S.cfg.nMels, frames]) };
    const out = await S.session.run(feeds);
    const t2 = performance.now();
    return { db: db, logits: out.logits.data, melMs: t1 - t0, modelMs: t2 - t1 };
  }

  function topK(logits, k) {
    const idx = Array.from(logits, (_, i) => i).sort((a, b) => logits[b] - logits[a]).slice(0, k);
    return idx.map(i => ({ i: i, p: sigmoid(logits[i]) }));
  }
  function dbfs(x) { let e = 0; for (let i = 0; i < x.length; i++) e += x[i] * x[i]; return 10 * Math.log10(Math.max(e / x.length, 1e-12)); }

  function render(rows) {
    const box = $('results'); box.textContent = '';
    rows.forEach(r => {
      const card = el('div', 'card');
      const head = el('div', 'head');
      head.appendChild(el('strong', '', r.start.toFixed(0) + '–' + (r.start + S.cfg.nSamples / S.cfg.sr).toFixed(0) + ' s'));
      head.appendChild(el('span', 'muted', 'level ' + r.level.toFixed(0) + ' dBFS · mel ' + r.melMs.toFixed(0) + ' ms · model ' + r.modelMs.toFixed(0) + ' ms'));
      card.appendChild(head);
      const top = topK(r.logits, 3);
      card.appendChild(el('div', top[0].p >= 0.5 ? 'verdict yes' : 'verdict no',
        top[0].p >= 0.5 ? 'Detected: ' + S.names[top[0].i] : 'No confident detection (best guess below)'));
      if (r.level < -60) card.appendChild(el('div', 'warn', 'Very quiet recording: move closer to the bird or raise the volume.'));
      top.forEach(t => {
        const line = el('div', 'row');
        line.appendChild(el('span', 'name', S.names[t.i] + ' (' + S.classes[t.i] + ')'));
        const bar = el('span', 'bar'); const fill = el('span', 'fill');
        fill.style.width = Math.round(t.p * 100) + '%'; bar.appendChild(fill); line.appendChild(bar);
        line.appendChild(el('span', 'pct', Math.round(t.p * 100) + '%'));
        card.appendChild(line);
      });
      box.appendChild(card);
    });
  }

  async function analyze(samples, sr) {
    setBusy(true);
    try {
      setStatus('Resampling to ' + S.cfg.sr + ' Hz…'); await tick();
      const x = sr === S.cfg.sr ? samples : BirdDSP.resample(samples, sr, S.cfg.sr);
      const wins = BirdDSP.splitWindows(x, S.cfg.nSamples, S.cfg.sr);
      if (!wins.length) { setStatus('Need at least 1 second of audio.', 'err'); return; }
      const rows = [];
      for (let i = 0; i < wins.length; i++) {
        setStatus('Analysing window ' + (i + 1) + ' of ' + wins.length + '…'); await tick();
        const r = await scoreWindow(wins[i].data);
        r.start = wins[i].start; r.level = dbfs(wins[i].data); rows.push(r);
      }
      render(rows);
      setStatus('Done: ' + wins.length + ' window' + (wins.length > 1 ? 's' : '') + ' analysed on this device.', 'ok');
    } catch (e) { setStatus('Error: ' + e.message, 'err'); }
    finally { setBusy(false); }
  }

  function toMono(buf) {
    const n = buf.length, out = new Float32Array(n), ch = buf.numberOfChannels;
    for (let c = 0; c < ch; c++) { const d = buf.getChannelData(c); for (let i = 0; i < n; i++) out[i] += d[i] / ch; }
    return out;
  }

  async function onFile(file) {
    if (!file) return;
    setBusy(true); setStatus('Decoding ' + file.name + '…');
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      let ctx; try { ctx = new AC({ sampleRate: S.cfg.sr }); } catch (e) { ctx = new AC(); }
      const buf = await ctx.decodeAudioData(await file.arrayBuffer());
      const mono = toMono(buf), sr = buf.sampleRate; ctx.close();
      setBusy(false); await analyze(mono, sr);
    } catch (e) { setBusy(false); setStatus('Could not read that audio file: ' + e.message, 'err'); }
  }

  const WORKLET = "class Rec extends AudioWorkletProcessor{process(i){const c=i[0]&&i[0][0];if(c)this.port.postMessage(c.slice(0));return true;}}registerProcessor('rec',Rec);";

  async function record(seconds) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { setStatus('Microphone access needs a secure (https) page.', 'err'); return; }
    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = new AC(); ctx.resume();                       // created inside the tap so iOS allows audio
    setBusy(true); let stream;
    try {
      // Browsers apply voice processing by default; it would suppress birdsong, so switch it all off.
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
      if (!ctx.audioWorklet) throw new Error('This browser has no AudioWorklet support.');
      await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' })));
      const src = ctx.createMediaStreamSource(stream), node = new AudioWorkletNode(ctx, 'rec');
      const mute = ctx.createGain(); mute.gain.value = 0;
      const chunks = []; node.port.onmessage = e => chunks.push(e.data);
      src.connect(node); node.connect(mute); mute.connect(ctx.destination);
      const t0 = performance.now();
      while ((performance.now() - t0) / 1000 < seconds) {
        setStatus('Recording… ' + Math.max(0, seconds - (performance.now() - t0) / 1000).toFixed(0) + ' s left', 'rec'); await new Promise(r => setTimeout(r, 200));
      }
      const sr = ctx.sampleRate;
      stream.getTracks().forEach(t => t.stop()); node.disconnect(); ctx.close();
      const total = chunks.reduce((a, c) => a + c.length, 0), samples = new Float32Array(total);
      let o = 0; chunks.forEach(c => { samples.set(c, o); o += c.length; });
      setBusy(false); await analyze(samples, sr);
    } catch (e) {
      if (stream) stream.getTracks().forEach(t => t.stop());
      try { ctx.close(); } catch (_) {}
      setBusy(false); setStatus('Recording failed: ' + e.message, 'err');
    }
  }

  async function selfTest() {
    setBusy(true); const out = $('results'); out.textContent = '';
    const card = el('div', 'card'); out.appendChild(card);
    const line = (t, cls) => card.appendChild(el('div', cls || 'row', t));
    try {
      setStatus('Running self-test…'); await tick();
      const f = await Promise.all(['fixture_wave.bin', 'fixture_wave48.bin', 'fixture_mel_db.bin', 'fixture_logits.bin'].map(n => fetchF32('data/' + n)));
      const wave = f[0], wave48 = f[1], wantDb = f[2], wantLogits = f[3];
      const r = await scoreWindow(wave);
      let mx = 0; for (let i = 0; i < wantDb.length; i++) mx = Math.max(mx, Math.abs(r.db[i] - wantDb[i]));
      let lx = 0; for (let i = 0; i < wantLogits.length; i++) lx = Math.max(lx, Math.abs(r.logits[i] - wantLogits[i]));
      const top = a => a.reduce((b, v, i) => v > a[b] ? i : b, 0);
      const same = top(r.logits) === top(wantLogits);
      line('Spectrogram vs librosa: max difference ' + mx.toFixed(4) + ' dB → ' + (mx < 0.05 ? 'PASS' : 'FAIL'), mx < 0.05 ? 'verdict yes' : 'verdict no');
      line('Model output vs Python: max logit difference ' + lx.toFixed(5) + ' → ' + (lx < 0.05 ? 'PASS' : 'FAIL'), lx < 0.05 ? 'verdict yes' : 'verdict no');
      line('Top-1 species identical to Python: ' + (same ? 'PASS' : 'FAIL') + ' (' + S.names[top(r.logits)] + ')', same ? 'verdict yes' : 'verdict no');
      const rs = BirdDSP.resample(wave48, 48000, S.cfg.sr).subarray(0, S.cfg.nSamples);
      const r2 = await scoreWindow(rs);
      let sum = 0; for (let i = 0; i < wantDb.length; i++) sum += Math.abs(r2.db[i] - wantDb[i]);
      const mean = sum / wantDb.length, same2 = top(r2.logits) === top(wantLogits);
      line('Via a 48 kHz recording (browser resampler): mean spectrogram difference ' + mean.toFixed(3) + ' dB, same top-1: ' + (same2 ? 'yes' : 'NO') + ' → ' + (mean < 0.1 && same2 ? 'PASS' : 'CHECK'), mean < 0.1 && same2 ? 'verdict yes' : 'verdict no');
      line('Speed on this device: spectrogram ' + r.melMs.toFixed(0) + ' ms + model ' + r.modelMs.toFixed(0) + ' ms per 5 s window', 'muted');
      setStatus('Self-test finished.', 'ok');
    } catch (e) { line('Self-test error: ' + e.message, 'verdict no'); setStatus('Self-test failed.', 'err'); }
    finally { setBusy(false); }
  }

  function wire() {
    $('btnRec').addEventListener('click', () => record(parseInt($('dur').value, 10)));
    $('btnFile').addEventListener('click', () => $('file').click());
    $('file').addEventListener('change', e => { const f = e.target.files && e.target.files[0]; onFile(f); e.target.value = ''; });
    $('btnTest').addEventListener('click', selfTest);
    setBusy(true); init();
  }
  window.BirdApp = { init: init, analyze: analyze, selfTest: selfTest, S: S };
  if (typeof document !== 'undefined' && document.getElementById('btnRec')) wire();
})();
