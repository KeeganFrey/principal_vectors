/* ============================================================
   InstrumentChart — drop-in interactive SVG line chart
   ------------------------------------------------------------
   ONE file, ZERO dependencies. Injects its own CSS + fonts.
   Solid + dashed series · animated draw-in · hover tooltip +
   crosshair + axis tag · legend toggle · X/Y zoom + pan.

   USAGE:
     <div id="myChart"></div>
     <script src="instrument-chart.js"></script>
     <script>
       new InstrumentChart('#myChart', {
         xLabel:'Iteration →', yLabel:'Performance (%)',
         xDomain:[0,24], yDomain:[0,100],
         series:[
           { name:'Throughput', color:'#2563eb', dash:false,
             points:[{x:0,y:22},{x:1,y:30}, ...] },
           { name:'Baseline',   color:'#dc2626', dash:true,
             points:[ ... ] }
         ]
       });
     </script>

   The instrument theme (framed tinted plot, chip readouts,
   replay button, integer X axis, Y-grid density) is applied
   automatically — no theme object required. Anything you pass
   in `theme` still overrides the defaults.

   GESTURES: scroll = X zoom · shift+scroll = Y zoom ·
   ctrl/cmd+scroll = Y-grid density · drag = pan · −/＋/Reset/↻.
   ============================================================ */
(function (global) {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';

  // ---- one-time asset injection: fonts + component CSS ----
  function injectAssets() {
    if (document.getElementById('instrument-chart-css')) return;
    // IBM Plex fonts (optional — falls back to system fonts offline)
    if (!document.querySelector('link[data-instrument-chart-font]')) {
      var l = document.createElement('link');
      l.rel = 'stylesheet';
      l.setAttribute('data-instrument-chart-font', '');
      l.href = 'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap';
      document.head.appendChild(l);
    }
    var css = [
      ".pc-wrap{position:relative;width:100%;font-family:'IBM Plex Sans',system-ui,sans-serif;}",
      ".pc-wrap .pc-legend{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:16px;}",
      ".pc-wrap .pc-leg{display:inline-flex;align-items:center;gap:8px;cursor:pointer;border:1px solid #e2e8f0;background:#fff;border-radius:6px;padding:7px 11px;font-family:inherit;font-size:12.5px;color:#334155;transition:opacity .15s,border-color .15s;}",
      ".pc-wrap .pc-leg:hover{border-color:#94a3b8;}",
      ".pc-wrap .pc-leg-off{opacity:.38;}",
      ".pc-wrap .pc-sw{width:18px;height:4px;border-radius:2px;flex:none;}",
      ".pc-wrap .pc-leg-name{white-space:nowrap;}",
      ".pc-wrap .pc-leg-val{font-family:'IBM Plex Mono',monospace;font-weight:600;color:#0f172a;min-width:34px;text-align:right;}",
      ".pc-wrap .pc-ctrl{position:absolute;top:6px;right:4px;display:flex;gap:5px;}",
      ".pc-wrap .pc-btn{font-family:'IBM Plex Mono',monospace;font-size:13px;line-height:1;width:26px;height:26px;border:1px solid #e2e8f0;background:#fff;color:#64748b;border-radius:5px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:.12s;padding:0;}",
      ".pc-wrap .pc-btn:hover{border-color:#94a3b8;color:#0f172a;background:#fbfcfd;}",
      ".pc-wrap .pc-btn-reset{width:auto;padding:0 9px;font-size:11px;letter-spacing:.03em;}",
      ".pc-wrap .pc-btn-replay{display:inline-flex;align-items:center;gap:5px;}",
      ".pc-wrap .pc-replay-ic{font-size:13px;line-height:1;}"
    ].join('\n');
    var st = document.createElement('style');
    st.id = 'instrument-chart-css';
    st.textContent = css;
    document.head.appendChild(st);
  }

  function el(tag, attrs) {
    var e = document.createElementNS(NS, tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function niceNum(range, round) {
    var exp = Math.floor(Math.log10(range || 1));
    var f = range / Math.pow(10, exp), nf;
    if (round) nf = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
    else nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
    return nf * Math.pow(10, exp);
  }
  function ticks(min, max, count) {
    if (min === max) { max = min + 1; }
    var step = niceNum(niceNum(max - min, false) / Math.max(1, count - 1), true);
    var start = Math.ceil(min / step) * step;
    var out = [];
    for (var v = start; v <= max + step * 1e-6; v += step) out.push(+v.toFixed(6));
    return out;
  }
  function bisect(arr, x) {
    var lo = 0, hi = arr.length - 1;
    if (x <= arr[0]) return 0;
    if (x >= arr[hi]) return hi;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (arr[mid] < x) lo = mid + 1; else hi = mid;
    }
    // nearest of lo-1, lo
    if (lo > 0 && Math.abs(arr[lo - 1] - x) <= Math.abs(arr[lo] - x)) return lo - 1;
    return lo;
  }

  // Instrument-panel theme baked in as defaults.
  var DEFAULTS = {
    height: 370,
    font: "'IBM Plex Sans', system-ui, sans-serif",
    mono: "'IBM Plex Mono', ui-monospace, monospace",
    axisColor: '#9ca3af',
    tickColor: '#64748b',
    labelColor: '#334155',
    titleColor: '#0f172a',
    gridColor: '#e6ebf1',
    gridX: true,
    gridY: true,
    plotBg: '#f4f7fa',    // tinted plot area
    frame: true,          // framed plot
    frameColor: '#cdd6e0',
    lineWidth: 2.4,
    pointR: 3.6,
    crosshair: true,
    crosshairColor: '#94a3b8',
    replayBtn: true,      // ↻ Replay button replays the draw-in animation
    tickIn: 0,            // inward tick mark length (px); 0 = none
    tickFont: 11,
    radius: 6,            // rounded plot frame / bg
    legend: 'chips',
    legendReadout: true,  // legend chips show the hovered value
    tooltip: false,       // axis tag + chip readouts instead of a floating box
    axisHoverLabel: true, // dark X-value tag pinned to the axis at the crosshair
    xInteger: true,       // integer X ticks (no duplicate labels; edge arrows)
    xTickTarget: 6,       // approx. number of labeled X ticks
    yTickCount: 6,        // initial Y gridline count (Ctrl/Cmd+wheel adjusts it)
    fmtX: function (v) { return (Math.round(v * 100) / 100) + ''; },
    fmtY: function (v) { return (Math.round(v * 100) / 100) + ''; }
  };

  function InstrumentChart(root, opts) {
    this.root = typeof root === 'string' ? document.querySelector(root) : root;
    this.o = {};
    for (var k in DEFAULTS) this.o[k] = DEFAULTS[k];
    if (opts.theme) for (var t in opts.theme) this.o[t] = opts.theme[t];
    this.xLabel = opts.xLabel || '';
    this.yLabel = opts.yLabel || '';
    this.type = opts.type || 'line';           // 'line' | 'bar'
    this.series = opts.series.map(function (s) {
      return { name: s.name, color: s.color, dash: !!s.dash, points: s.points, hidden: false };
    });

    // domains
    var xs = [], ys = [];
    this.series.forEach(function (s) { s.points.forEach(function (p) { xs.push(p.x); ys.push(p.y); }); });
    this.xExtent = opts.xDomain || [Math.min.apply(null, xs), Math.max.apply(null, xs)];
    if (opts.yDomain) { this.yDomain = opts.yDomain; }
    else {
      var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
      var padY = (ymax - ymin) * 0.08 || 1;
      this.yDomain = [ymin - padY, ymax + padY];
    }
    this.yExtent = [this.yDomain[0], this.yDomain[1]];
    this.view = { x0: this.xExtent[0], x1: this.xExtent[1], y0: this.yExtent[0], y1: this.yExtent[1] };
    this.yTickTarget = this.o.yTickCount || 5;
    this.uid = 'pc' + Math.random().toString(36).slice(2, 8);
    injectAssets();
    this._build();
    var self = this;
    this._ro = new ResizeObserver(function () { self.render(); });
    this._ro.observe(this.wrap);
    this.render();
    requestAnimationFrame(function () { self._animateIn(); });
  }

  InstrumentChart.prototype._build = function () {
    var o = this.o, self = this;
    var wrap = el('svg' === '' ? 'div' : 'div'); // wrapper div
    wrap = document.createElement('div');
    wrap.className = 'pc-wrap';
    wrap.style.position = 'relative';
    wrap.style.width = '100%';
    wrap.style.fontFamily = o.font;
    this.wrap = wrap;

    // legend
    var legend = document.createElement('div');
    legend.className = 'pc-legend';
    this.legendEl = legend;
    this._buildLegend();

    var svg = el('svg', { width: '100%', height: o.height, 'aria-label': 'line chart' });
    svg.style.display = 'block';
    svg.style.touchAction = 'none';
    svg.style.cursor = 'crosshair';
    this.svg = svg;

    var defs = el('defs');
    var clip = el('clipPath', { id: this.uid + '-clip' });
    this.clipRect = el('rect', { x: 0, y: 0, width: 0, height: 0 });
    clip.appendChild(this.clipRect);
    defs.appendChild(clip);
    svg.appendChild(defs);

    this.gBg = el('g'); svg.appendChild(this.gBg);
    this.gGrid = el('g'); svg.appendChild(this.gGrid);
    this.gAxes = el('g'); svg.appendChild(this.gAxes);
    this.gPlot = el('g', { 'clip-path': 'url(#' + this.uid + '-clip)' }); svg.appendChild(this.gPlot);
    this.gOver = el('g', { 'clip-path': 'url(#' + this.uid + '-clip)' }); svg.appendChild(this.gOver);
    this.gFrame = el('g'); svg.appendChild(this.gFrame);
    this.gHover = el('g'); svg.appendChild(this.gHover);

    // tooltip
    var tip = document.createElement('div');
    tip.className = 'pc-tip';
    tip.style.cssText = 'position:absolute;pointer-events:none;opacity:0;transition:opacity .12s;' +
      'background:#0f172a;color:#f8fafc;font:500 12px/1.5 ' + o.mono + ';' +
      'padding:9px 11px;border-radius:' + (o.radius ? 8 : 4) + 'px;white-space:nowrap;' +
      'box-shadow:0 6px 24px rgba(15,23,42,.22);z-index:5;transform:translate(-50%,-115%);';
    this.tip = tip;

    // controls (zoom)
    var ctrl = document.createElement('div');
    ctrl.className = 'pc-ctrl';
    ['−', '＋', 'Reset'].forEach(function (lab, i) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'pc-btn' + (i === 2 ? ' pc-btn-reset' : '');
      b.textContent = lab;
      b.addEventListener('click', function () {
        if (i === 0) self._zoomBy(1 / 0.7);
        else if (i === 1) self._zoomBy(0.7);
        else self._resetView();
      });
      ctrl.appendChild(b);
    });
    if (o.replayBtn) {
      var rb = document.createElement('button');
      rb.type = 'button';
      rb.className = 'pc-btn pc-btn-reset pc-btn-replay';
      rb.innerHTML = '<span class="pc-replay-ic">↻</span> Replay';
      rb.addEventListener('click', function () { self.replay(); });
      ctrl.appendChild(rb);
    }
    this.ctrl = ctrl;

    var stage = document.createElement('div');
    stage.style.position = 'relative';
    stage.appendChild(svg);
    stage.appendChild(tip);
    stage.appendChild(ctrl);

    wrap.appendChild(legend);
    wrap.appendChild(stage);
    this.root.appendChild(wrap);

    // interactions
    svg.addEventListener('mousemove', function (e) { self._hover(e); });
    svg.addEventListener('mouseleave', function () { self._hideHover(); });
    svg.addEventListener('wheel', function (e) { self._wheel(e); }, { passive: false });
    svg.addEventListener('pointerdown', function (e) { self._panStart(e); });
  };

  InstrumentChart.prototype._buildLegend = function () {
    var self = this, o = this.o;
    clear(this.legendEl);
    this.series.forEach(function (s, i) {
      var item = document.createElement('button');
      item.type = 'button';
      item.className = 'pc-leg' + (s.hidden ? ' pc-leg-off' : '');
      var sw = document.createElement('span');
      sw.className = 'pc-sw';
      sw.style.background = s.dash
        ? 'repeating-linear-gradient(90deg,' + s.color + ' 0 6px,transparent 6px 11px)'
        : s.color;
      var nm = document.createElement('span');
      nm.className = 'pc-leg-name';
      nm.textContent = s.name;
      item.appendChild(sw); item.appendChild(nm);
      if (o.legendReadout) {
        var rd = document.createElement('span');
        rd.className = 'pc-leg-val';
        rd.textContent = '—';
        item.appendChild(rd);
        s._readout = rd;
      }
      item.addEventListener('click', function () {
        s.hidden = !s.hidden;
        item.classList.toggle('pc-leg-off', s.hidden);
        self.render();
      });
      self.legendEl.appendChild(item);
    });
  };

  InstrumentChart.prototype._yTicksArr = function () {
    var y0 = this.view.y0, y1 = this.view.y1;
    return ticks(y0, y1, this.yTickTarget).filter(function (v) {
      return v >= y0 - 1e-9 && v <= y1 + 1e-9;
    });
  };
  InstrumentChart.prototype._xTicks = function () {
    var o = this.o, v0 = this.view.x0, v1 = this.view.x1, span = v1 - v0;
    if (!o.xInteger) {
      var arr = ticks(v0, v1, 7).filter(function (v) { return v >= v0 - 1e-9 && v <= v1 + 1e-9; });
      return { labeled: arr.map(function (v) { return { v: v, text: o.fmtX(v) }; }), minor: [], arrows: null };
    }
    var steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000, 2000, 5000];
    var step = steps[steps.length - 1];
    for (var i = 0; i < steps.length; i++) { if (span / steps[i] <= o.xTickTarget) { step = steps[i]; break; } }
    var labeled = [], start = Math.ceil((v0 - 1e-9) / step) * step;
    for (var v = start; v <= v1 + 1e-9; v += step) labeled.push({ v: v, text: '' + Math.round(v) });
    var minorStep = step >= 5 ? step / 5 : (step === 2 ? 1 : 0), minor = [];
    if (minorStep > 0 && span / minorStep <= 80) {
      var ms = Math.ceil((v0 - 1e-9) / minorStep) * minorStep;
      for (var m = ms; m <= v1 + 1e-9; m += minorStep) {
        if (Math.abs(m / step - Math.round(m / step)) < 1e-6) continue; // skip labeled positions
        minor.push(m);
      }
    }
    var arrows = labeled.length === 0
      ? { left: Math.floor(v0 / step) * step, right: Math.ceil(v1 / step) * step }
      : null;
    return { labeled: labeled, minor: minor, arrows: arrows };
  };

  InstrumentChart.prototype._geom = function () {
    var o = this.o;
    var W = this.wrap.clientWidth || 640;
    var H = o.height;
    var padL = this.yLabel ? 58 : 46;
    var padB = this.xLabel ? 46 : 30;
    var padT = 14, padR = 16;
    return {
      W: W, H: H,
      x: padL, y: padT,
      w: Math.max(10, W - padL - padR),
      h: Math.max(10, H - padT - padB)
    };
  };

  InstrumentChart.prototype._sx = function (v, g) {
    return g.x + (v - this.view.x0) / (this.view.x1 - this.view.x0) * g.w;
  };
  InstrumentChart.prototype._sy = function (v, g) {
    return g.y + (1 - (v - this.view.y0) / (this.view.y1 - this.view.y0)) * g.h;
  };

  InstrumentChart.prototype.render = function () {
    var o = this.o, g = this._geom();
    this._g = g;
    this.svg.setAttribute('viewBox', '0 0 ' + g.W + ' ' + g.H);
    // keep clip full unless intro is animating
    if (!this._introing) this.clipRect.setAttribute('width', g.w);
    this.clipRect.setAttribute('x', g.x);
    this.clipRect.setAttribute('y', g.y);
    this.clipRect.setAttribute('height', g.h);

    clear(this.gBg); clear(this.gGrid); clear(this.gAxes); clear(this.gPlot); clear(this.gFrame); clear(this.gHover);

    // plot background
    if (o.plotBg && o.plotBg !== 'none') {
      this.gBg.appendChild(el('rect', {
        x: g.x, y: g.y, width: g.w, height: g.h, rx: o.radius, fill: o.plotBg
      }));
    }

    var xticks = this._xTicks();
    var yt = this._yTicksArr();

    var self = this;
    // grid
    if (o.gridY) yt.forEach(function (v) {
      var y = self._sy(v, g);
      self.gGrid.appendChild(el('line', { x1: g.x, y1: y, x2: g.x + g.w, y2: y, stroke: o.gridColor, 'stroke-width': 1 }));
    });
    if (o.gridX) xticks.labeled.forEach(function (t) {
      var x = self._sx(t.v, g);
      self.gGrid.appendChild(el('line', { x1: x, y1: g.y, x2: x, y2: g.y + g.h, stroke: o.gridColor, 'stroke-width': 1 }));
    });

    // axes lines
    this.gAxes.appendChild(el('line', { x1: g.x, y1: g.y + g.h, x2: g.x + g.w, y2: g.y + g.h, stroke: o.axisColor, 'stroke-width': 1.25 }));
    this.gAxes.appendChild(el('line', { x1: g.x, y1: g.y, x2: g.x, y2: g.y + g.h, stroke: o.axisColor, 'stroke-width': 1.25 }));

    // ticks + labels
    yt.forEach(function (v) {
      var y = self._sy(v, g);
      if (o.tickIn) self.gAxes.appendChild(el('line', { x1: g.x, y1: y, x2: g.x + o.tickIn, y2: y, stroke: o.axisColor, 'stroke-width': 1.25 }));
      else self.gAxes.appendChild(el('line', { x1: g.x - 4, y1: y, x2: g.x, y2: y, stroke: o.axisColor, 'stroke-width': 1.25 }));
      var tx = self._txt(g.x - 8, y, o.fmtY(v), { anchor: 'end', baseline: 'middle', mono: true });
      self.gAxes.appendChild(tx);
    });
    xticks.labeled.forEach(function (t) {
      var x = self._sx(t.v, g);
      if (o.tickIn) self.gAxes.appendChild(el('line', { x1: x, y1: g.y + g.h, x2: x, y2: g.y + g.h - o.tickIn, stroke: o.axisColor, 'stroke-width': 1.25 }));
      else self.gAxes.appendChild(el('line', { x1: x, y1: g.y + g.h, x2: x, y2: g.y + g.h + 5, stroke: o.axisColor, 'stroke-width': 1.25 }));
      self.gAxes.appendChild(self._txt(x, g.y + g.h + 17, t.text, { anchor: 'middle', baseline: 'auto', mono: true }));
    });
    xticks.minor.forEach(function (v) {
      var x = self._sx(v, g), len = 3.5;
      if (o.tickIn) self.gAxes.appendChild(el('line', { x1: x, y1: g.y + g.h, x2: x, y2: g.y + g.h - len, stroke: o.axisColor, 'stroke-width': 1 }));
      else self.gAxes.appendChild(el('line', { x1: x, y1: g.y + g.h, x2: x, y2: g.y + g.h + len, stroke: o.axisColor, 'stroke-width': 1 }));
    });
    if (xticks.arrows) {
      self.gAxes.appendChild(self._txt(g.x + 3, g.y + g.h + 17, '‹ ' + Math.round(xticks.arrows.left), { anchor: 'start', baseline: 'auto', mono: true }));
      self.gAxes.appendChild(self._txt(g.x + g.w - 3, g.y + g.h + 17, Math.round(xticks.arrows.right) + ' ›', { anchor: 'end', baseline: 'auto', mono: true }));
    }

    // axis titles
    if (this.xLabel) {
      this.gAxes.appendChild(this._txt(g.x + g.w / 2, g.H - 4, this.xLabel, { anchor: 'middle', baseline: 'auto', title: true }));
    }
    if (this.yLabel) {
      var yt2 = this._txt(14, g.y + g.h / 2, this.yLabel, { anchor: 'middle', baseline: 'middle', title: true });
      yt2.setAttribute('transform', 'rotate(-90,14,' + (g.y + g.h / 2) + ')');
      this.gAxes.appendChild(yt2);
    }

    // series — grouped bars or lines+bands
    if (this.type === 'bar') {
      this._drawBars(g);
    } else {
      // ±1σ error bands behind the lines
      this.series.forEach(function (s) {
        if (s.hidden || !s.points.some(function (p) { return p.e != null; })) return;
        var up = '', dn = '';
        s.points.forEach(function (p, i) {
          var e = p.e || 0;
          var X = self._sx(p.x, g);
          up += (i ? 'L' : 'M') + X.toFixed(2) + ' ' + self._sy(p.y + e, g).toFixed(2) + ' ';
        });
        for (var i = s.points.length - 1; i >= 0; i--) {
          var p = s.points[i], e = p.e || 0;
          dn += 'L' + self._sx(p.x, g).toFixed(2) + ' ' + self._sy(p.y - e, g).toFixed(2) + ' ';
        }
        self.gPlot.appendChild(el('path', {
          d: up + dn + 'Z', fill: s.color, 'fill-opacity': 0.12, stroke: 'none'
        }));
      });
      // lines
      this.series.forEach(function (s) {
        if (s.hidden) return;
        var d = '', started = false;
        s.points.forEach(function (p) {
          var X = self._sx(p.x, g), Y = self._sy(p.y, g);
          d += (started ? 'L' : 'M') + X.toFixed(2) + ' ' + Y.toFixed(2) + ' ';
          started = true;
        });
        var path = el('path', {
          d: d, fill: 'none', stroke: s.color,
          'stroke-width': o.lineWidth,
          'stroke-linejoin': 'round', 'stroke-linecap': 'round'
        });
        if (s.dash) path.setAttribute('stroke-dasharray', '8 5');
        self.gPlot.appendChild(path);
      });
    }

    // frame
    if (o.frame) {
      this.gFrame.appendChild(el('rect', {
        x: g.x, y: g.y, width: g.w, height: g.h, rx: o.radius,
        fill: 'none', stroke: o.frameColor, 'stroke-width': 1.25
      }));
    }
  };

  /* ---- diagonal hatch fill (re-used for `dash` series in bar mode) ---- */
  InstrumentChart.prototype._hatch = function (color) {
    var id = this.uid + '-h' + color.replace(/[^a-z0-9]/gi, '');
    if (!document.getElementById(id)) {
      var defs = this.svg.querySelector('defs');
      var pat = el('pattern', {
        id: id, width: 6, height: 6, patternUnits: 'userSpaceOnUse',
        patternTransform: 'rotate(45)'
      });
      pat.appendChild(el('rect', { width: 6, height: 6, fill: color, 'fill-opacity': 0.18 }));
      pat.appendChild(el('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: color, 'stroke-width': 2.4 }));
      defs.appendChild(pat);
    }
    return 'url(#' + id + ')';
  };

  /* ---- grouped bar chart with ±1σ error whiskers ---- */
  InstrumentChart.prototype._drawBars = function (g) {
    var self = this, o = this.o;
    var vis = this.series.filter(function (s) { return !s.hidden; });
    // category x-positions = the union of integer x-values across series
    var catSet = {};
    this.series.forEach(function (s) { s.points.forEach(function (p) { catSet[p.x] = true; }); });
    var cats = Object.keys(catSet).map(Number).sort(function (a, b) { return a - b; });
    if (!cats.length || !vis.length) { this._barGeom = []; return; }

    var unit = g.w / (this.view.x1 - this.view.x0);   // px per 1 data unit
    var clusterW = Math.min(unit * 0.8, 86);          // width of one category's cluster
    var barW = clusterW / vis.length;
    var baseY = this._sy(this.view.y0, g);
    var geom = [];

    cats.forEach(function (cx) {
      var center = self._sx(cx, g);
      var x0 = center - clusterW / 2;
      var items = [];
      vis.forEach(function (s, k) {
        var p = null;
        for (var i = 0; i < s.points.length; i++) if (s.points[i].x === cx) { p = s.points[i]; break; }
        if (!p) return;
        var bx = x0 + k * barW;
        var topY = self._sy(p.y, g);
        var h = Math.max(0, baseY - topY);
        self.gPlot.appendChild(el('rect', {
          x: bx.toFixed(2), y: topY.toFixed(2), width: Math.max(1, barW - 2).toFixed(2),
          height: h.toFixed(2), rx: 2,
          fill: s.dash ? self._hatch(s.color) : s.color,
          stroke: s.color, 'stroke-width': s.dash ? 1 : 0, 'fill-opacity': s.dash ? 1 : 0.9
        }));
        // ±1σ whisker
        if (p.e != null) {
          var mid = bx + (barW - 2) / 2;
          var yHi = self._sy(p.y + p.e, g), yLo = self._sy(p.y - p.e, g);
          var cap = Math.min(5, (barW - 2) / 2.4);
          var wk = el('path', {
            d: 'M' + mid + ' ' + yHi + 'V' + yLo +
               'M' + (mid - cap) + ' ' + yHi + 'H' + (mid + cap) +
               'M' + (mid - cap) + ' ' + yLo + 'H' + (mid + cap),
            stroke: '#0f172a', 'stroke-width': 1.2, 'stroke-opacity': 0.72, fill: 'none'
          });
          self.gPlot.appendChild(wk);
        }
        items.push({ s: s, p: p, mid: bx + (barW - 2) / 2 });
      });
      geom.push({ x: cx, center: center, x0: x0, x1: x0 + clusterW, items: items });
    });
    this._barGeom = geom;
  };

  InstrumentChart.prototype._txt = function (x, y, str, opt) {
    opt = opt || {};
    var o = this.o;
    var t = el('text', {
      x: x, y: y,
      'text-anchor': opt.anchor || 'start',
      fill: opt.title ? o.labelColor : o.tickColor,
      'font-family': opt.mono ? o.mono : o.font,
      'font-size': opt.title ? 12 : o.tickFont,
      'font-weight': opt.title ? 600 : 400
    });
    if (opt.baseline === 'middle') t.setAttribute('dominant-baseline', 'middle');
    if (opt.title) { t.setAttribute('letter-spacing', '.02em'); }
    t.textContent = str;
    return t;
  };

  /* ---------- intro animation: sweep reveal ---------- */
  InstrumentChart.prototype._animateIn = function () {
    var self = this, g = this._g, dur = 900, t0 = null;
    // If the tab is hidden, rAF won't fire — just show the chart fully.
    if (typeof document !== 'undefined' && document.hidden) {
      this._introing = false;
      this.clipRect.setAttribute('width', g.w);
      return;
    }
    this._introing = true;
    this.clipRect.setAttribute('width', 0);
    function step(ts) {
      if (t0 === null) t0 = ts;
      var p = Math.min(1, (ts - t0) / dur);
      var e = 1 - Math.pow(1 - p, 3); // ease-out cubic
      self.clipRect.setAttribute('width', g.w * e);
      if (p < 1) requestAnimationFrame(step);
      else { self._introing = false; self.clipRect.setAttribute('width', g.w); }
    }
    requestAnimationFrame(step);
  };
  InstrumentChart.prototype.replay = function () {
    this.render();        // make sure geometry/clip are current
    this._animateIn();    // re-run the left-to-right sweep reveal
  };

  /* ---------- hover ---------- */
  InstrumentChart.prototype._hover = function (e) {
    var g = this._g, o = this.o;
    var rect = this.svg.getBoundingClientRect();
    var scale = g.W / rect.width;
    var mx = (e.clientX - rect.left) * scale;
    if (mx < g.x || mx > g.x + g.w) { this._hideHover(); return; }
    if (this.type === 'bar') { this._hoverBar(mx, g); return; }
    var xVal = this.view.x0 + (mx - g.x) / g.w * (this.view.x1 - this.view.x0);

    clear(this.gOver);
    clear(this.gHover);
    var any = false, rows = [];
    // crosshair
    var snapX = null;
    var self = this;
    this.series.forEach(function (s) {
      if (s.hidden) { if (s._readout) s._readout.textContent = '—'; return; }
      var xs = s.points.map(function (p) { return p.x; });
      var idx = bisect(xs, xVal);
      var p = s.points[idx];
      if (p.x < self.view.x0 - 1e-9 || p.x > self.view.x1 + 1e-9) { if (s._readout) s._readout.textContent = '—'; return; }
      snapX = p.x;
      var cx = self._sx(p.x, g), cy = self._sy(p.y, g);
      self.gOver.appendChild(el('circle', {
        cx: cx, cy: cy, r: o.pointR + 1.5, fill: '#fff', stroke: s.color, 'stroke-width': 2
      }));
      rows.push({ s: s, val: p.y });
      if (s._readout) s._readout.textContent = o.fmtY(p.y);
      any = true;
    });

    if (snapX !== null && o.crosshair) {
      var lx = this._sx(snapX, g);
      this.gOver.insertBefore(el('line', {
        x1: lx, y1: g.y, x2: lx, y2: g.y + g.h,
        stroke: o.crosshairColor, 'stroke-width': 1, 'stroke-dasharray': '3 3'
      }), this.gOver.firstChild);
    }

    if (any) {
      if (o.axisHoverLabel) {
        var ax = this._sx(snapX, g);
        var lbl = o.fmtX(snapX);
        var halfw = 8 + lbl.length * 4.2;
        var cx2 = Math.max(g.x + halfw, Math.min(g.x + g.w - halfw, ax));
        var by = g.y + g.h;
        this.gHover.appendChild(el('rect', {
          x: cx2 - halfw, y: by + 5, width: halfw * 2, height: 18, rx: 3, fill: '#0f172a'
        }));
        var lt = el('text', {
          x: cx2, y: by + 15, 'text-anchor': 'middle', 'dominant-baseline': 'middle',
          fill: '#f8fafc', 'font-family': o.mono, 'font-size': 11, 'font-weight': 600
        });
        lt.textContent = lbl;
        this.gHover.appendChild(lt);
      }
      if (o.tooltip) {
        var html = '<div style="opacity:.7;font-size:10.5px;letter-spacing:.04em;margin-bottom:5px">' +
          (this.xLabel ? this.xLabel.replace(/[→↑].*$/, '').trim() : 'x') + ' = ' + o.fmtX(snapX) + '</div>';
        rows.forEach(function (r) {
          var sw = r.s.dash
            ? 'background:repeating-linear-gradient(90deg,' + r.s.color + ' 0 4px,transparent 4px 7px)'
            : 'background:' + r.s.color;
          html += '<div style="display:flex;align-items:center;gap:7px;margin:2px 0">' +
            '<span style="width:11px;height:3px;border-radius:1px;' + sw + '"></span>' +
            '<span style="opacity:.85">' + r.s.name + '</span>' +
            '<span style="margin-left:auto;font-weight:600">' + o.fmtY(r.val) + '</span></div>';
        });
        this.tip.innerHTML = html;
        this.tip.style.left = this._sx(snapX, g) / scale + 'px';
        this.tip.style.top = (g.y) / scale + 'px';
        this.tip.style.opacity = '1';
      }
    }
  };
  InstrumentChart.prototype._hoverBar = function (mx, g) {
    var o = this.o, geom = this._barGeom || [];
    clear(this.gOver); clear(this.gHover);
    // nearest category to the cursor
    var cat = null, best = Infinity;
    geom.forEach(function (c) {
      var d = Math.abs((c.x0 + c.x1) / 2 - mx);
      if (d < best) { best = d; cat = c; }
    });
    this.series.forEach(function (s) { if (s._readout) s._readout.textContent = '—'; });
    if (!cat) return;
    // soft column highlight behind the hovered cluster
    var pad = 6;
    this.gOver.appendChild(el('rect', {
      x: cat.x0 - pad, y: g.y, width: (cat.x1 - cat.x0) + pad * 2, height: g.h,
      fill: '#0f172a', 'fill-opacity': 0.04
    }));
    cat.items.forEach(function (it) {
      if (it.s._readout) {
        it.s._readout.textContent = o.fmtY(it.p.y) + (it.p.e != null ? ' ±' + o.fmtY(it.p.e) : '');
      }
    });
    // x-axis category tag
    var lbl = o.fmtX(cat.x), halfw = 8 + lbl.length * 4.2;
    var cx2 = Math.max(g.x + halfw, Math.min(g.x + g.w - halfw, cat.center));
    var by = g.y + g.h;
    this.gHover.appendChild(el('rect', { x: cx2 - halfw, y: by + 5, width: halfw * 2, height: 18, rx: 3, fill: '#0f172a' }));
    var lt = el('text', {
      x: cx2, y: by + 15, 'text-anchor': 'middle', 'dominant-baseline': 'middle',
      fill: '#f8fafc', 'font-family': o.mono, 'font-size': 11, 'font-weight': 600
    });
    lt.textContent = lbl;
    this.gHover.appendChild(lt);
  };

  InstrumentChart.prototype._hideHover = function () {
    clear(this.gOver);
    clear(this.gHover);
    this.tip.style.opacity = '0';
    var o = this.o;
    this.series.forEach(function (s) { if (s._readout) s._readout.textContent = '—'; });
  };

  /* ---------- zoom / pan ---------- */
  InstrumentChart.prototype._wheel = function (e) {
    e.preventDefault();
    var g = this._g;
    var rect = this.svg.getBoundingClientRect();
    var scale = g.W / rect.width;
    // Ctrl / Cmd + wheel -> adjust Y-grid density
    if (e.ctrlKey || e.metaKey) {
      this._setDensity(this.yTickTarget + (e.deltaY < 0 ? 1 : -1));
      return;
    }
    var factor = e.deltaY < 0 ? 0.85 : 1 / 0.85;
    // Shift + wheel -> zoom the Y axis (anchored at cursor)
    if (e.shiftKey) {
      var my = (e.clientY - rect.top) * scale;
      var fracY = Math.min(1, Math.max(0, (my - g.y) / g.h));
      this._zoomYAt(fracY, factor);
      return;
    }
    // default -> zoom the X axis
    var mx = (e.clientX - rect.left) * scale;
    var frac = Math.min(1, Math.max(0, (mx - g.x) / g.w));
    this._zoomAt(frac, factor);
  };
  InstrumentChart.prototype._zoomBy = function (factor) { this._zoomAt(0.5, factor); };
  InstrumentChart.prototype._zoomAt = function (frac, factor) {
    var v = this.view, span = v.x1 - v.x0;
    var anchor = v.x0 + frac * span;
    var full = this.xExtent[1] - this.xExtent[0];
    var newSpan = Math.min(full, Math.max(full * 0.04, span * factor));
    var x0 = anchor - frac * newSpan;
    var x1 = x0 + newSpan;
    if (x0 < this.xExtent[0]) { x0 = this.xExtent[0]; x1 = x0 + newSpan; }
    if (x1 > this.xExtent[1]) { x1 = this.xExtent[1]; x0 = x1 - newSpan; }
    this.view.x0 = x0; this.view.x1 = x1;
    this.render();
  };
  InstrumentChart.prototype._zoomYAt = function (fracTop, factor) {
    var v = this.view, span = v.y1 - v.y0;
    var anchor = v.y0 + (1 - fracTop) * span;        // data value under the cursor
    var full = this.yExtent[1] - this.yExtent[0];
    var newSpan = Math.min(full, Math.max(full * 0.04, span * factor));
    var y0 = anchor - (1 - fracTop) * newSpan;
    var y1 = y0 + newSpan;
    if (y0 < this.yExtent[0]) { y0 = this.yExtent[0]; y1 = y0 + newSpan; }
    if (y1 > this.yExtent[1]) { y1 = this.yExtent[1]; y0 = y1 - newSpan; }
    this.view.y0 = y0; this.view.y1 = y1;
    this.render();
  };
  InstrumentChart.prototype._setDensity = function (n) {
    n = Math.max(3, Math.min(24, n));
    if (n === this.yTickTarget) return;
    this.yTickTarget = n;
    this.render();
  };
  InstrumentChart.prototype._resetView = function () {
    this.view = { x0: this.xExtent[0], x1: this.xExtent[1], y0: this.yExtent[0], y1: this.yExtent[1] };
    this.render();
  };
  InstrumentChart.prototype._panStart = function (e) {
    var g = this._g, self = this;
    var rect = this.svg.getBoundingClientRect();
    var scale = g.W / rect.width;
    var xSpan = this.view.x1 - this.view.x0, ySpan = this.view.y1 - this.view.y0;
    var xZoom = xSpan < (this.xExtent[1] - this.xExtent[0]) - 1e-9;
    var yZoom = ySpan < (this.yExtent[1] - this.yExtent[0]) - 1e-9;
    if (!xZoom && !yZoom) return; // nothing to pan
    var startX = e.clientX, startY = e.clientY;
    var v0 = { x0: this.view.x0, x1: this.view.x1, y0: this.view.y0, y1: this.view.y1 };
    var uppX = xSpan / g.w * scale, uppY = ySpan / g.h * scale;
    this.svg.style.cursor = 'grabbing';
    this.tip.style.opacity = '0';
    clear(this.gHover);
    function move(ev) {
      if (xZoom) {
        var dx = (ev.clientX - startX) * uppX;
        var x0 = v0.x0 - dx, x1 = v0.x1 - dx;
        if (x0 < self.xExtent[0]) { x0 = self.xExtent[0]; x1 = x0 + xSpan; }
        if (x1 > self.xExtent[1]) { x1 = self.xExtent[1]; x0 = x1 - xSpan; }
        self.view.x0 = x0; self.view.x1 = x1;
      }
      if (yZoom) {
        var dy = (ev.clientY - startY) * uppY;
        var y0 = v0.y0 + dy, y1 = v0.y1 + dy;
        if (y0 < self.yExtent[0]) { y0 = self.yExtent[0]; y1 = y0 + ySpan; }
        if (y1 > self.yExtent[1]) { y1 = self.yExtent[1]; y0 = y1 - ySpan; }
        self.view.y0 = y0; self.view.y1 = y1;
      }
      self.render();
    }
    function up() {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      self.svg.style.cursor = 'crosshair';
    }
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  global.InstrumentChart = InstrumentChart;
  global.PerfChart = InstrumentChart; // back-compat alias
})(window);

