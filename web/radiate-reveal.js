/* ============================================================
   Radiate reveal — full-screen "Try the models" panel.
   Opens with a circular iris wipe, hosts a draw pad wired to
   real in-browser inference (pv-infer.js) and a gallery of
   layer-0 principal-vector reconstructions.
   ============================================================ */
(function () {
  'use strict';
  var body = document.body;
  var root = document.documentElement;

  /* ---------------- open / close ---------------- */
  var panel = document.getElementById('panel');
  var closeBtn = document.getElementById('panelClose');

  function openPanel(originEl) {
    var el = originEl || document.getElementById('floatBtn');
    var r = el.getBoundingClientRect();
    root.style.setProperty('--cx', (r.left + r.width / 2) + 'px');
    root.style.setProperty('--cy', (r.top + r.height / 2) + 'px');
    body.classList.add('open');
    closeBtn.focus({ preventScroll: true });
    if (!galleryBuilt) buildGallery();
  }
  function closePanel() { body.classList.remove('open'); }

  document.addEventListener('click', function (e) {
    var opener = e.target.closest('[data-open]');
    if (opener) openPanel(opener);
  });
  closeBtn.addEventListener('click', closePanel);
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closePanel(); });

  /* ============================================================
     DRAW PAD + real inference
     ============================================================ */
  var pad = document.getElementById('pad');
  var ctx = pad.getContext('2d', { willReadFrequently: true });
  var hint = document.getElementById('padHint');
  var modelSel = document.getElementById('modelSel');
  var predsEl = document.getElementById('preds');
  var verdict = document.getElementById('verdict');
  var drawing = false, hasInk = false, last = null, busy = false, queued = false;

  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.strokeStyle = '#f4f4f2'; ctx.lineWidth = 20;

  function resetPad() {
    ctx.fillStyle = '#0d0e11'; ctx.fillRect(0, 0, 240, 240);
    hasInk = false; hint.style.opacity = 1;
    renderPreds(null);
    verdict.innerHTML = '<b>—</b>draw a character to begin';
  }
  function pos(e) {
    var r = pad.getBoundingClientRect();
    return { x: (e.clientX - r.left) * 240 / r.width, y: (e.clientY - r.top) * 240 / r.height };
  }
  pad.addEventListener('pointerdown', function (e) {
    drawing = true; hasInk = true; hint.style.opacity = 0; last = pos(e);
    pad.setPointerCapture(e.pointerId);
  });
  pad.addEventListener('pointermove', function (e) {
    if (!drawing) return;
    var p = pos(e);
    ctx.beginPath(); ctx.moveTo(last.x, last.y); ctx.lineTo(p.x, p.y); ctx.stroke();
    last = p;
  });
  pad.addEventListener('pointerup', function () { if (drawing) { drawing = false; predict(); } });
  pad.addEventListener('pointerleave', function () { if (drawing) { drawing = false; predict(); } });
  document.getElementById('clearBtn').addEventListener('click', resetPad);
  modelSel.addEventListener('change', function () { if (hasInk) predict(); });

  function renderPreds(top) {
    predsEl.innerHTML = '';
    if (!top) {
      for (var d = 0; d < 10; d++) {
        var row = document.createElement('div'); row.className = 'pred';
        row.innerHTML = '<span class="dnum">·</span><span class="bar"><i style="width:0%"></i></span><span class="pct">—</span>';
        predsEl.appendChild(row);
      }
      return;
    }
    top.forEach(function (r, i) {
      var row = document.createElement('div');
      row.className = 'pred' + (i === 0 ? ' top' : '');
      var pct = (r.p * 100).toFixed(0);
      row.innerHTML = '<span class="dnum">' + escapeHtml(r.label) + '</span>' +
        '<span class="bar"><i style="width:' + pct + '%"></i></span>' +
        '<span class="pct">' + pct + '%</span>';
      predsEl.appendChild(row);
    });
  }
  function escapeHtml(s) { return s.replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }

  function predict() {
    if (!hasInk) { renderPreds(null); return; }
    if (busy) { queued = true; return; }
    busy = true;
    var key = modelSel.value;
    verdict.innerHTML = '<b>…</b>running ' + (PV.MODELS[key].mode === 'principal' ? 'principal' : 'linear') + ' model';
    PV.predict(key, ctx).then(function (res) {
      busy = false;
      if (queued) { queued = false; predict(); return; }
      if (!res) { renderPreds(null); verdict.innerHTML = '<b>—</b>draw a character to begin'; return; }
      var top = PV.topK(res.probs, res.labels, Math.min(8, res.labels.length));
      renderPreds(top);
      verdict.innerHTML = '<b>' + escapeHtml(top[0].label) + '</b>' + (top[0].p * 100).toFixed(1) + '% confident';
    }).catch(function (err) {
      busy = false;
      verdict.innerHTML = '<b>!</b>' + (err && err.message ? err.message : 'inference failed');
      console.error(err);
    });
  }

  /* ============================================================
     GALLERY — layer-0 principal-vector reconstructions
     ============================================================ */
  var galleryBuilt = false;
  var GALLERY_NET = { mnist: 'gallery/best_mnist_deep_ln_drop', emnist: 'gallery/best_emnist_deep_ln_drop' };
  var DISP = 42;                 // displayed tile px
  var NOTE_PRINCIPAL = 'Because the first hidden layer reads raw pixels, each of its principal vectors is itself a point in image space. Each row is one neuron; each tile is one of the distinct activation patterns that neuron was simplified down to.';
  var NOTE_DIVIDED = 'Each principal vector is the contribution W⊙x, so dividing it elementwise by the neuron’s weight row recovers the implied input x — the image-space pattern that would drive this neuron toward that principal vector. Pixels the neuron barely weights are dropped to gray.';

  /* render the prebuilt sprite (raw layer-0 principal vectors) */
  function renderSprite(net, man, host) {
    host.innerHTML = '';
    var url = GALLERY_NET[net] + '/sprite.png';
    var sizeW = man.cols * DISP, sizeH = man.rows * DISP;
    man.neurons.forEach(function (nrow, r) {
      var strip = document.createElement('div'); strip.className = 'gal-row';
      var lab = document.createElement('span'); lab.className = 'gal-lab';
      lab.textContent = 'n' + nrow.neuron;
      strip.appendChild(lab);
      for (var c = 0; c < man.cols; c++) {
        var cell = document.createElement('div'); cell.className = 'gal-cell';
        cell.style.width = cell.style.height = DISP + 'px';
        cell.style.backgroundImage = 'url(' + url + ')';
        cell.style.backgroundSize = sizeW + 'px ' + sizeH + 'px';
        cell.style.backgroundPosition = '-' + (c * DISP) + 'px -' + (r * DISP) + 'px';
        cell.title = 'neuron ' + nrow.neuron + ' · principal vector ' + c;
        strip.appendChild(cell);
      }
      host.appendChild(strip);
    });
  }

  /* render principal-vector ÷ weight reconstructions, computed in-browser
     from the same blobs the inference engine uses. p ≈ W⊙x for layer 0,
     so p/W ≈ x recovers the implied input image. */
  function renderDivided(net, man, host) {
    return PV.loadNet(net)
      .then(function (rec) { return PV.ensurePrincipal(rec); })
      .then(function (rec) {
        host.innerHTML = '';
        var f32 = rec.f32, i8 = rec.i8, L = rec.meta.layers[0];
        var ii = L.in, wo = L.wOff / 4, g = L.g;
        // int8 offset where each neuron's principal-vector block begins
        var pvAt = [], acc = L.pvOff;
        for (var nn = 0; nn < g.length; nn++) { pvAt[nn] = acc; acc += g[nn] * ii; }

        man.neurons.forEach(function (nrow) {
          var n = nrow.neuron;
          var strip = document.createElement('div'); strip.className = 'gal-row';
          var lab = document.createElement('span'); lab.className = 'gal-lab';
          lab.textContent = 'n' + n;
          strip.appendChild(lab);

          var wbase = wo + n * ii, k, maxW = 0;
          for (k = 0; k < ii; k++) { var aw = Math.abs(f32[wbase + k]); if (aw > maxW) maxW = aw; }
          var eps = 0.05 * maxW || 1e-6;            // ignore pixels the neuron barely reads
          var count = Math.min(nrow.count, g[n]);

          for (var c = 0; c < man.cols; c++) {
            var cell = document.createElement('canvas'); cell.className = 'gal-cell';
            cell.width = 28; cell.height = 28;
            cell.style.width = cell.style.height = DISP + 'px';
            if (c < count) {
              var poff = pvAt[n] + c * ii, d = new Float32Array(ii), mags = new Float32Array(ii);
              for (k = 0; k < ii; k++) {
                var w = f32[wbase + k];
                d[k] = Math.abs(w) > eps ? i8[poff + k] / w : 0;
                mags[k] = Math.abs(d[k]);
              }
              // robust contrast: stretch by the 98th-percentile magnitude so a
              // single near-zero-weight pixel can't blow out the whole tile.
              var sorted = Array.prototype.slice.call(mags).sort(function (a, b) { return a - b; });
              var scale = sorted[Math.floor(0.98 * (ii - 1))] || sorted[ii - 1] || 1;
              var cx = cell.getContext('2d'), im = cx.createImageData(28, 28);
              for (k = 0; k < ii; k++) {
                var val = 0.5 + 0.5 * d[k] / scale;
                if (val < 0) val = 0; else if (val > 1) val = 1;
                var gray = (val * 255) | 0;
                im.data[k * 4] = gray; im.data[k * 4 + 1] = gray; im.data[k * 4 + 2] = gray; im.data[k * 4 + 3] = 255;
              }
              cx.putImageData(im, 0, 0);
              cell.title = 'neuron ' + n + ' · principal vector ' + c + ' ÷ weights';
            } else {
              cell.title = '—';
            }
            strip.appendChild(cell);
          }
          host.appendChild(strip);
        });
      });
  }

  function buildGallery() {
    galleryBuilt = true;
    var host = document.getElementById('gallery');
    var sel = document.getElementById('gallerySel');
    var divToggle = document.getElementById('galDivide');
    var note = document.getElementById('galleryNote');
    if (!host) return;

    function load(net) {
      var divided = divToggle && divToggle.checked;
      if (note) note.textContent = divided ? NOTE_DIVIDED : NOTE_PRINCIPAL;
      host.innerHTML = '<div class="gal-loading">' +
        (divided ? 'reconstructing inputs…' : 'loading reconstructions…') + '</div>';
      fetch(GALLERY_NET[net] + '/sprite.json').then(function (r) { return r.json(); }).then(function (man) {
        if (divided) return renderDivided(net, man, host);
        renderSprite(net, man, host);
      }).catch(function () { host.innerHTML = '<div class="gal-loading">gallery unavailable</div>'; });
    }
    if (sel) sel.addEventListener('change', function () { load(sel.value); });
    if (divToggle) divToggle.addEventListener('change', function () { load(sel ? sel.value : 'mnist'); });
    load(sel ? sel.value : 'mnist');
  }

  /* ---------------- init ---------------- */
  resetPad();
})();
