/* ============================================================
   pv-infer.js — real in-browser inference for the principal-
   vector explainer. Mirrors scripts/validate_web_assets.py
   (the numpy reference) byte-for-byte against the same blobs:

     models/<net>/meta.json    layer shapes + offsets + labels
     models/<net>/linear.f32   weights, biases, layernorm, rowsums
     models/<net>/principal.i8  int8 unit-normalized principal vectors

   Two forward modes per network:
     linear     — y = LN(W·x + b) -> ReLU, head -> softmax
     principal  — each neuron's contribution W_n⊙x is snapped to its
                  nearest principal vector (cosine); pre-activation
                  becomes Σ(p*) + b, then the usual LN/ReLU proceed.
   ============================================================ */
window.PV = (function () {
  'use strict';

  // logical model -> network dir + default mode
  var MODELS = {
    'linear-mnist':    { net: 'mnist',  mode: 'linear',    label: 'Linear baseline · MNIST' },
    'principal-mnist': { net: 'mnist',  mode: 'principal', label: 'Principal (k=100) · MNIST' },
    'linear-emnist':   { net: 'emnist', mode: 'linear',    label: 'Linear baseline · EMNIST-bal' },
    'principal-emnist':{ net: 'emnist', mode: 'principal', label: 'Principal (k=100) · EMNIST-bal' }
  };
  var NET_DIR = {
    mnist:  'models/best_mnist_deep_ln_drop',
    emnist: 'models/best_emnist_deep_ln_drop'
  };

  var cache = {};   // net -> { meta, f32, i8(optional), i8promise }

  function loadNet(net) {
    if (cache[net] && cache[net].ready) return Promise.resolve(cache[net]);
    if (cache[net] && cache[net].loading) return cache[net].loading;
    var dir = NET_DIR[net];
    var rec = { net: net };
    rec.loading = Promise.all([
      fetch(dir + '/meta.json').then(function (r) { return r.json(); }),
      fetch(dir + '/linear.f32').then(function (r) { return r.arrayBuffer(); })
    ]).then(function (out) {
      rec.meta = out[0];
      rec.f32 = new Float32Array(out[1]);
      rec.ready = true;
      cache[net] = rec;
      return rec;
    });
    cache[net] = rec;
    return rec.loading;
  }

  // principal blob is ~13MB — load only when a principal mode is used
  function ensurePrincipal(rec) {
    if (rec.i8) return Promise.resolve(rec);
    if (rec.i8promise) return rec.i8promise;
    rec.i8promise = fetch(NET_DIR[rec.net] + '/principal.i8')
      .then(function (r) { return r.arrayBuffer(); })
      .then(function (buf) { rec.i8 = new Int8Array(buf); return rec; });
    return rec.i8promise;
  }

  // ---- math ----
  function layerNorm(x, f32, ngOff, nbOff, n) {
    var mean = 0, i;
    for (i = 0; i < n; i++) mean += x[i];
    mean /= n;
    var v = 0, d;
    for (i = 0; i < n; i++) { d = x[i] - mean; v += d * d; }
    v /= n;
    var inv = 1 / Math.sqrt(v + 1e-5);
    var go = ngOff / 4, bo = nbOff / 4;
    for (i = 0; i < n; i++) x[i] = f32[go + i] * (x[i] - mean) * inv + f32[bo + i];
  }

  function forwardLinear(rec, x) {
    var f32 = rec.f32, layers = rec.meta.layers, L;
    for (var li = 0; li < layers.length; li++) {
      L = layers[li];
      var oi = L.out, ii = L.in, wo = L.wOff / 4, bo = L.bOff / 4;
      var out = new Float32Array(oi);
      for (var n = 0; n < oi; n++) {
        var s = f32[bo + n], base = wo + n * ii;
        for (var k = 0; k < ii; k++) s += f32[base + k] * x[k];
        out[n] = s;
      }
      if (L.ngOff !== undefined) layerNorm(out, f32, L.ngOff, L.nbOff, oi);
      if (L.relu) for (var r = 0; r < oi; r++) if (out[r] < 0) out[r] = 0;
      x = out;
    }
    return x;
  }

  function forwardPrincipal(rec, x) {
    var f32 = rec.f32, i8 = rec.i8, layers = rec.meta.layers, L;
    for (var li = 0; li < layers.length; li++) {
      L = layers[li];
      var oi = L.out, ii = L.in, wo = L.wOff / 4, bo = L.bOff / 4;
      var rs = L.rsOff / 4, g = L.g, pvCur = L.pvOff, rsCur = 0;
      var out = new Float32Array(oi);
      var h = new Float32Array(ii);
      for (var n = 0; n < oi; n++) {
        var wbase = wo + n * ii, k;
        for (k = 0; k < ii; k++) h[k] = f32[wbase + k] * x[k];   // contribution
        var gc = g[n], best = -Infinity, bestG = 0, gi, off = pvCur;
        for (gi = 0; gi < gc; gi++) {
          var dot = 0;
          for (k = 0; k < ii; k++) dot += i8[off + k] * h[k];     // ∝ cosine
          if (dot > best) { best = dot; bestG = gi; }
          off += ii;
        }
        out[n] = f32[rs + rsCur + bestG] + f32[bo + n];
        pvCur += gc * ii; rsCur += gc;
      }
      if (L.ngOff !== undefined) layerNorm(out, f32, L.ngOff, L.nbOff, oi);
      if (L.relu) for (var r = 0; r < oi; r++) if (out[r] < 0) out[r] = 0;
      x = out;
    }
    return x;
  }

  function softmax(z) {
    var m = -Infinity, i;
    for (i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
    var s = 0, out = new Float32Array(z.length);
    for (i = 0; i < z.length; i++) { out[i] = Math.exp(z[i] - m); s += out[i]; }
    for (i = 0; i < z.length; i++) out[i] /= s;
    return out;
  }

  /* ---- canvas -> normalized 28×28, MNIST-style centering ---- */
  function preprocess(ctx, mean, std) {
    var W = 240, S = 240;
    var data = ctx.getImageData(0, 0, W, S).data;
    // 1) intensity grid (ink is light on dark bg) + bounding box
    var minx = W, miny = S, maxx = -1, maxy = -1, x, y;
    var g = new Float32Array(W * S);
    for (y = 0; y < S; y++) for (x = 0; x < W; x++) {
      var v = data[(y * W + x) * 4] / 255;   // R channel
      g[y * W + x] = v;
      if (v > 0.12) {
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
      }
    }
    if (maxx < 0) return null;               // no ink
    // 2) scale bbox into a 20×20 box on a 28×28 temp canvas
    var src = document.createElement('canvas'); src.width = W; src.height = S;
    var sctx = src.getContext('2d');
    sctx.putImageData(ctx.getImageData(0, 0, W, S), 0, 0);
    var bw = maxx - minx + 1, bh = maxy - miny + 1;
    var scale = 20 / Math.max(bw, bh);
    var dw = Math.max(1, Math.round(bw * scale)), dh = Math.max(1, Math.round(bh * scale));
    var tmp = document.createElement('canvas'); tmp.width = 28; tmp.height = 28;
    var tctx = tmp.getContext('2d');
    tctx.fillStyle = '#000'; tctx.fillRect(0, 0, 28, 28);
    tctx.imageSmoothingEnabled = true;
    var dx = Math.round((28 - dw) / 2), dy = Math.round((28 - dh) / 2);
    tctx.drawImage(src, minx, miny, bw, bh, dx, dy, dw, dh);
    var td = tctx.getImageData(0, 0, 28, 28).data;
    // 3) center of mass -> shift to (14,14)
    var raw = new Float32Array(784), sum = 0, cx = 0, cy = 0, i;
    for (y = 0; y < 28; y++) for (x = 0; x < 28; x++) {
      var iv = td[(y * 28 + x) * 4] / 255;
      raw[y * 28 + x] = iv; sum += iv; cx += iv * x; cy += iv * y;
    }
    var shx = sum ? Math.round(14 - cx / sum) : 0, shy = sum ? Math.round(14 - cy / sum) : 0;
    var outv = new Float32Array(784);
    for (y = 0; y < 28; y++) for (x = 0; x < 28; x++) {
      var sxp = x - shx, syp = y - shy;
      var val = (sxp >= 0 && sxp < 28 && syp >= 0 && syp < 28) ? raw[syp * 28 + sxp] : 0;
      outv[y * 28 + x] = (val - mean) / std;
    }
    return outv;
  }

  /* ---- public: run one prediction ---- */
  function predict(modelKey, ctx) {
    var spec = MODELS[modelKey];
    return loadNet(spec.net).then(function (rec) {
      var x = preprocess(ctx, rec.meta.inMean, rec.meta.inStd);
      if (!x) return null;
      var run = function () {
        var logits = spec.mode === 'principal'
          ? forwardPrincipal(rec, x) : forwardLinear(rec, x);
        return { probs: softmax(logits), labels: rec.meta.labels };
      };
      if (spec.mode === 'principal') return ensurePrincipal(rec).then(run);
      return run();
    });
  }

  function topK(probs, labels, k) {
    var idx = Array.prototype.map.call(probs, function (p, i) { return i; });
    idx.sort(function (a, b) { return probs[b] - probs[a]; });
    return idx.slice(0, k).map(function (i) {
      return { label: labels[i], p: probs[i], idx: i };
    });
  }

  return {
    MODELS: MODELS, NET_DIR: NET_DIR,
    loadNet: loadNet, ensurePrincipal: ensurePrincipal,
    predict: predict, topK: topK
  };
})();
