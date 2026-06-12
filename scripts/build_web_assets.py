"""Build static web assets for the principal-vector explainer page.

Reads the committed research artifacts in this repo and emits everything the
browser page (``web/index.html``) needs:

    web/data/figN.json          chart data for InstrumentChart (figures 1-9)
    web/models/<net>/meta.json  network + principal-substitution manifest
    web/models/<net>/linear.f32  float32 weights, biases, layernorm, rowsums
    web/models/<net>/principal.i8  int8 unit-normalized principal vectors
    web/gallery/<net>/sprite.png   layer-0 principal-vector reconstructions
    web/gallery/<net>/sprite.json  sprite layout manifest

Run from anywhere:  python scripts/build_web_assets.py
"""
from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # principal_vectors/ (repo root)
FIGS = ROOT / "draft_figures"
MODELS = ROOT / "models"
WEB = ROOT / "web"

# MNIST/EMNIST normalization used at training time.
IN_MEAN, IN_STD = 0.1307, 0.3081

# Plot palette (kept consistent with the page's blue accent).
PALETTE = ["#2563eb", "#16a34a", "#d97706", "#7c3aed", "#dc2626", "#0891b2"]

CONFIG_LABEL = {"plain": "plain", "drop": "dropout",
                "ln": "layernorm", "ln_drop": "ln+dropout"}
# Preferred config when a figure has to pick one network variant to show.
CONFIG_PREF = ["ln_drop", "drop", "ln", "plain"]
# Canonical color per config, used by every figure that denotes a config by
# color so the legend reads the same across the whole page.
CONFIG_ORDER = ["plain", "ln", "drop", "ln_drop"]
CONFIG_COLOR = {"plain": "#2563eb", "ln": "#16a34a",
                "drop": "#d97706", "ln_drop": "#7c3aed"}

# EMNIST-balanced 47-class label -> character (standard mapping).
EMNIST_CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
MNIST_CHARS = list("0123456789")


# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #
def read_csv(name: str) -> list[dict]:
    with open(FIGS / f"{name}.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def f(row: dict, key: str):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def series(name, color, points, dash=False):
    return {"name": name, "color": color, "dash": dash, "points": points}


def ept(x, row, mean_field, scale=100.0):
    """Point dict {x, y} with an optional error bar `e` taken from the
    matching ``*_std`` column (``acc_norm_mean`` -> ``acc_norm_std`` etc.).
    Zero / missing std (e.g. single-seed runs) is left off."""
    p = {"x": x, "y": f(row, mean_field) * scale}
    e = f(row, mean_field.replace("_mean", "_std"))
    if e == e and e > 0:                      # not NaN and nonzero
        p["e"] = e * scale
    return p


def flat_y(s):
    """All y-values across series, expanded by ±e so error bars stay in view."""
    out = []
    for ser in s:
        for p in ser["points"]:
            out.append(p["y"])
            if "e" in p:
                out.append(p["y"] + p["e"])
                out.append(p["y"] - p["e"])
    return out


def domain(values, pad_frac=0.06, lo=None, hi=None):
    vs = [v for v in values if v == v]            # drop NaN
    a, b = min(vs), max(vs)
    pad = (b - a) * pad_frac or max(abs(b), 1) * 0.05
    return [lo if lo is not None else a - pad,
            hi if hi is not None else b + pad]


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")))


def pick_config(rows, dataset, arch):
    have = {r["config"] for r in rows
            if r["dataset"] == dataset and r["arch"] == arch}
    for c in CONFIG_PREF:
        if c in have:
            return c
    return sorted(have)[0]


# --------------------------------------------------------------------------- #
#  figures 1-9  ->  web/data/figN.json
# --------------------------------------------------------------------------- #
def fig1():
    """Headline accuracy: deep MNIST/EMNIST, linear vs principal, per config."""
    rows = read_csv("figure1_headline_accuracy")
    configs = ["plain", "drop", "ln", "ln_drop"]
    xt = [CONFIG_LABEL[c] for c in configs]

    def pts(dataset, field):
        out = []
        for i, c in enumerate(configs):
            r = next((r for r in rows if r["dataset"] == dataset
                      and r["arch"] == "deep" and r["config"] == c), None)
            if r:
                out.append(ept(i, r, field))
        return out

    s = [
        series("MNIST · linear", PALETTE[0], pts("mnist", "test_acc_none_mean")),
        series("MNIST · principal", PALETTE[0],
               pts("mnist", "test_acc_hidden_mean"), dash=True),
        series("EMNIST · linear", PALETTE[1], pts("emnist", "test_acc_none_mean")),
        series("EMNIST · principal", PALETTE[1],
               pts("emnist", "test_acc_hidden_mean"), dash=True),
    ]
    ally = flat_y(s)
    return {"type": "bar",
            "xLabel": "Regularization (deep model) →", "yLabel": "Test accuracy (%)",
            "xDomain": [-0.5, len(configs) - 0.5], "yDomain": domain(ally, lo=0),
            "xTickLabels": xt, "series": s}


def fig2():
    """Shallow MNIST: accuracy vs # principal neurons substituted (normalized)."""
    rows = read_csv("figure2_shallow_mnist_neuron_substitution")
    combos = [("fc1", "index"), ("fc1", "random"),
              ("fc2", "index"), ("fc2", "random")]
    color = {"fc1": PALETTE[0], "fc2": PALETTE[2]}
    s = []
    for layer, ordering in combos:
        pts = [ept(int(float(r["k"])), r, "acc_norm_mean")
               for r in rows if r["layer"] == layer and r["ordering"] == ordering]
        pts.sort(key=lambda p: p["x"])
        if pts:
            s.append(series(f"{layer} · {ordering}", color[layer], pts,
                            dash=(ordering == "random")))
    ally = flat_y(s)
    allx = [p["x"] for ser in s for p in ser["points"]]
    return {"xLabel": "Principal neurons substituted →",
            "yLabel": "Accuracy vs k=0 (%)",
            "xDomain": [0, max(allx)], "yDomain": domain(ally),
            "integerX": True, "series": s}


def _subset_fig(name):
    rows = read_csv(name)
    configs = sorted({r["config"] for r in rows}, key=lambda c: CONFIG_PREF.index(c)
                     if c in CONFIG_PREF else 9)
    s = []
    for i, c in enumerate(configs):
        pts = [ept(int(float(r["subset_size"])), r, "delta_mean")
               for r in rows if r["config"] == c]
        pts.sort(key=lambda p: p["x"])
        s.append(series(CONFIG_LABEL.get(c, c),
                        CONFIG_COLOR.get(c, PALETTE[i % len(PALETTE)]), pts))
    ally = flat_y(s)
    allx = [p["x"] for ser in s for p in ser["points"]]
    return {"xLabel": "# EMNIST classes in subset →",
            "yLabel": "Linear − principal acc (%)",
            "xDomain": [min(allx), max(allx)], "yDomain": domain(ally, lo=0),
            "integerX": True, "series": s}


def fig3():
    return _subset_fig("figure3_shallow_emnist_label_subsets")


def fig4():
    return _subset_fig("figure4_deep_emnist_label_subsets")


def fig5():
    """Per-layer substitution for one network: single vs prefix vs baseline."""
    rows = read_csv("figure5_per_layer_substitution")
    dataset, arch = "mnist", "deep"
    cfg = pick_config(rows, dataset, arch)
    sub = [r for r in rows if r["dataset"] == dataset
           and r["arch"] == arch and r["config"] == cfg]

    def pts(kind):
        p = [ept(int(float(r["layer"])), r, "test_mean")
             for r in sub if r["kind"] == kind]
        p.sort(key=lambda q: q["x"])
        return p

    single, prefix = pts("single"), pts("prefix")
    base_row = next((r for r in sub if r["kind"] == "baseline"), None)
    base = f(base_row, "test_mean") * 100 if base_row else None
    xs = [p["x"] for p in single + prefix]
    s = [series("single layer", PALETTE[0], single),
         series("prefix (cumulative)", PALETTE[2], prefix, dash=True)]
    if base is not None:
        s.append(series("all-linear baseline", PALETTE[4],
                        [{"x": min(xs), "y": base}, {"x": max(xs), "y": base}],
                        dash=True))
    ally = flat_y(s)
    return {"xLabel": f"Layer index ({dataset.upper()} deep · {CONFIG_LABEL[cfg]}) →",
            "yLabel": "Test accuracy (%)",
            "xDomain": [min(xs), max(xs)], "yDomain": domain(ally, lo=0),
            "integerX": True, "series": s}


# Config -> (config, color) for the per-config overlay charts (figs 5/6/7 a-d),
# in canonical order. single/principal = solid, prefix/linear = dashed.
BREAKDOWN_CONFIGS = [(c, CONFIG_COLOR[c]) for c in CONFIG_ORDER]


def fig5_breakdown(dataset, arch):
    """Per-layer substitution overlaid for every regularization config of one
    (dataset, arch). Each config contributes a `single` (solid) and a `prefix`
    (dashed) line so all four variants share one chart."""
    rows = read_csv("figure5_per_layer_substitution")
    s = []
    for cfg, color in BREAKDOWN_CONFIGS:
        sub = [r for r in rows if r["dataset"] == dataset
               and r["arch"] == arch and r["config"] == cfg]
        if not sub:
            continue
        label = CONFIG_LABEL.get(cfg, cfg)
        for kind, dash in (("single", False), ("prefix", True)):
            pts = [ept(int(float(r["layer"])), r, "test_mean")
                   for r in sub if r["kind"] == kind]
            pts.sort(key=lambda q: q["x"])
            if pts:
                s.append(series(f"{label} · {kind}", color, pts, dash=dash))
    ally = flat_y(s)
    allx = [p["x"] for ser in s for p in ser["points"]]
    return {"xLabel": f"Layer index ({dataset.upper()} {arch}) →",
            "yLabel": "Test accuracy (%)",
            "xDomain": [min(allx), max(allx)], "yDomain": domain(ally, lo=0),
            "integerX": True, "series": s}


def fig6():
    """Principal accuracy declines as the base model trains longer."""
    rows = read_csv("figure6_training_epochs")
    s = []
    for i, dataset in enumerate(("mnist", "emnist")):
        cfg = pick_config(rows, dataset, "deep")
        sub = [r for r in rows if r["dataset"] == dataset
               and r["arch"] == "deep" and r["config"] == cfg]
        sub.sort(key=lambda r: int(float(r["epoch_n"])))
        prin = [ept(int(float(r["epoch_n"])), r, "test_acc_hidden_mean") for r in sub]
        lin = [ept(int(float(r["epoch_n"])), r, "test_acc_none_mean") for r in sub]
        s.append(series(f"{dataset.upper()} · principal", PALETTE[i * 2], prin))
        s.append(series(f"{dataset.upper()} · linear", PALETTE[i * 2], lin, dash=True))
    ally = flat_y(s)
    allx = [p["x"] for ser in s for p in ser["points"]]
    return {"xLabel": "Training epochs of base model →",
            "yLabel": "Test accuracy (%)",
            "xDomain": [min(allx), max(allx)], "yDomain": domain(ally),
            "integerX": True, "series": s}


def fig6_breakdown(dataset, arch):
    """Training-epoch sweep overlaid for every regularization config of one
    (dataset, arch). Each config gets a `principal` (solid, all-hidden
    substituted) and a `linear` baseline (dashed) line."""
    rows = read_csv("figure6_training_epochs")
    s = []
    for cfg, color in BREAKDOWN_CONFIGS:
        sub = [r for r in rows if r["dataset"] == dataset
               and r["arch"] == arch and r["config"] == cfg]
        if not sub:
            continue
        sub.sort(key=lambda r: int(float(r["epoch_n"])))
        label = CONFIG_LABEL.get(cfg, cfg)
        prin = [ept(int(float(r["epoch_n"])), r, "test_acc_hidden_mean") for r in sub]
        lin = [ept(int(float(r["epoch_n"])), r, "test_acc_none_mean") for r in sub]
        s.append(series(f"{label} · principal", color, prin))
        s.append(series(f"{label} · linear", color, lin, dash=True))
    ally = flat_y(s)
    allx = [p["x"] for ser in s for p in ser["points"]]
    return {"xLabel": f"Training epochs of base model ({dataset.upper()} {arch}) →",
            "yLabel": "Test accuracy (%)",
            "xDomain": [min(allx), max(allx)], "yDomain": domain(ally),
            "integerX": True, "series": s}


def fig7_breakdown(dataset, arch):
    """Group-count sweep (K = 1..20) overlaid for every regularization config of
    one (dataset, arch). Each config gets a `principal` (solid) and a `linear`
    baseline (dashed, flat) line."""
    rows = read_csv("figure7_groups_1_to_20")
    s = []
    for cfg, color in BREAKDOWN_CONFIGS:
        sub = [r for r in rows if r["dataset"] == dataset
               and r["arch"] == arch and r["config"] == cfg]
        if not sub:
            continue
        sub.sort(key=lambda r: int(float(r["K"])))
        label = CONFIG_LABEL.get(cfg, cfg)
        prin = [ept(int(float(r["K"])), r, "prin_mean") for r in sub]
        lin = [ept(int(float(r["K"])), r, "linear_mean") for r in sub]
        s.append(series(f"{label} · principal", color, prin))
        s.append(series(f"{label} · linear", color, lin, dash=True))
    ally = flat_y(s)
    allx = [p["x"] for ser in s for p in ser["points"]]
    return {"xLabel": f"# principal groups (K) ({dataset.upper()} {arch}) →",
            "yLabel": "Test accuracy (%)",
            "xDomain": [min(allx), max(allx)], "yDomain": domain(ally, lo=0),
            "integerX": True, "series": s}


def _groups_fig(name, datasets, per_config):
    rows = read_csv(name)
    s = []
    if per_config:
        dataset = datasets[0]
        configs = sorted({r["config"] for r in rows if r["dataset"] == dataset
                          and r["arch"] == "deep"},
                         key=lambda c: CONFIG_PREF.index(c) if c in CONFIG_PREF else 9)
        for i, c in enumerate(configs):
            sub = [r for r in rows if r["dataset"] == dataset
                   and r["arch"] == "deep" and r["config"] == c]
            sub.sort(key=lambda r: int(float(r["K"])))
            pts = [ept(int(float(r["K"])), r, "prin_mean") for r in sub]
            s.append(series(CONFIG_LABEL.get(c, c),
                            CONFIG_COLOR.get(c, PALETTE[i % len(PALETTE)]), pts))
    else:
        for i, dataset in enumerate(datasets):
            cfg = pick_config(rows, dataset, "deep")
            sub = [r for r in rows if r["dataset"] == dataset
                   and r["arch"] == "deep" and r["config"] == cfg]
            sub.sort(key=lambda r: int(float(r["K"])))
            pts = [ept(int(float(r["K"])), r, "prin_mean") for r in sub]
            s.append(series(f"{dataset.upper()} · principal",
                            PALETTE[i * 2], pts))
    ally = flat_y(s)
    allx = [p["x"] for ser in s for p in ser["points"]]
    return {"xLabel": "# principal groups (K) →", "yLabel": "Test accuracy (%)",
            "xDomain": [min(allx), max(allx)], "yDomain": domain(ally, lo=0),
            "integerX": True, "series": s}


def fig7():
    return _groups_fig("figure7_groups_1_to_20", ["mnist", "emnist"], per_config=False)


def fig8():
    return _groups_fig("figure8_groups_beyond_20_mnist", ["mnist"], per_config=True)


def fig9():
    return _groups_fig("figure9_groups_beyond_20", ["emnist"], per_config=True)


def build_figures():
    builders = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]
    for i, fn in enumerate(builders, start=1):
        write_json(WEB / "data" / f"fig{i}.json", fn())
        print(f"  fig{i}.json")
    # per-config breakdowns: one chart per (dataset, arch) for figs 5/6/7.
    families = {"a": ("mnist", "shallow"), "b": ("mnist", "deep"),
                "c": ("emnist", "shallow"), "d": ("emnist", "deep")}
    for builder, base in ((fig5_breakdown, "fig5"), (fig6_breakdown, "fig6"),
                          (fig7_breakdown, "fig7")):
        for suffix, (ds, arch) in families.items():
            name = f"{base}{suffix}"
            write_json(WEB / "data" / f"{name}.json", builder(ds, arch))
            print(f"  {name}.json")


# --------------------------------------------------------------------------- #
#  models  ->  inference manifest + binary blobs
# --------------------------------------------------------------------------- #
def load_safetensors(path):
    t = {}
    with safe_open(str(path), framework="numpy") as fh:
        for k in fh.keys():
            t[k] = fh.get_tensor(k)
    return t


def build_model(net_dir: Path, dataset: str):
    name = net_dir.name
    st_path = next(net_dir.glob("*.safetensors"))
    tensors = load_safetensors(st_path)

    hidden_idx = sorted(int(k.split(".")[1]) for k in tensors
                        if k.startswith("linears.") and k.endswith(".weight"))
    layer_defs = []
    for i in hidden_idx:
        layer_defs.append({
            "w": tensors[f"linears.{i}.weight"], "b": tensors[f"linears.{i}.bias"],
            "ng": tensors.get(f"norms.{i}.weight"), "nb": tensors.get(f"norms.{i}.bias"),
        })
    layer_defs.append({"w": tensors["head.weight"], "b": tensors["head.bias"],
                       "ng": None, "nb": None})

    f32 = bytearray()      # weights, biases, norms, rowsums
    i8 = bytearray()       # int8 unit-normalized principal vectors
    meta_layers = []

    for li, ld in enumerate(layer_defs):
        w = ld["w"].astype(np.float32)          # (out, in)
        b = ld["b"].astype(np.float32)
        out_n, in_n = w.shape

        w_off = len(f32); f32 += w.tobytes()
        b_off = len(f32); f32 += b.tobytes()
        entry = {"in": int(in_n), "out": int(out_n),
                 "wOff": w_off, "bOff": b_off,
                 "relu": li != len(layer_defs) - 1}
        if ld["ng"] is not None:
            ng = ld["ng"].astype(np.float32); nb = ld["nb"].astype(np.float32)
            entry["ngOff"] = len(f32); f32 += ng.tobytes()
            entry["nbOff"] = len(f32); f32 += nb.tobytes()

        # principal vectors for this layer: one .npy per neuron
        lp = net_dir / f"layer{li}"
        gcounts, pv_off, rs_off = [], len(i8), None
        rs_off = None
        rowsum_buf = bytearray()
        for n in range(out_n):
            f_npy = next(lp.glob(f"neuron{n}_*_principles_mean.npy"))
            pv = np.load(f_npy).astype(np.float32)         # (G, in)
            norm = np.linalg.norm(pv, axis=1, keepdims=True)
            unit = pv / np.clip(norm, 1e-12, None)
            q = np.clip(np.round(unit * 127.0), -127, 127).astype(np.int8)
            i8 += q.tobytes()
            rowsum_buf += pv.sum(axis=1).astype(np.float32).tobytes()
            gcounts.append(int(pv.shape[0]))
        rs_off = len(f32); f32 += rowsum_buf
        entry["pvOff"] = pv_off          # int8 offset (rows are `in` wide)
        entry["rsOff"] = rs_off          # float32 rowsum offset
        entry["g"] = gcounts             # clusters per neuron
        meta_layers.append(entry)
        print(f"    {name} layer{li}: {out_n}x{in_n}, "
              f"{sum(gcounts)} principal vectors")

    chars = MNIST_CHARS if dataset == "mnist" else EMNIST_CHARS
    meta = {
        "name": name, "dataset": dataset,
        "numClasses": len(chars), "labels": chars,
        "inMean": IN_MEAN, "inStd": IN_STD,
        "layers": meta_layers,
    }
    out = WEB / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "linear.f32").write_bytes(bytes(f32))
    (out / "principal.i8").write_bytes(bytes(i8))
    write_json(out / "meta.json", meta)
    print(f"  {name}: linear.f32={len(f32)/1e3:.0f}KB  "
          f"principal.i8={len(i8)/1e6:.1f}MB")
    return name


# --------------------------------------------------------------------------- #
#  gallery: layer-0 principal vectors rendered as 28x28 reconstructions
# --------------------------------------------------------------------------- #
def build_gallery(net_dir: Path, n_neurons=24, n_principles=8, tile=28):
    name = net_dir.name
    lp = net_dir / "layer0"
    rows = []
    for n in range(n_neurons):
        try:
            f_npy = next(lp.glob(f"neuron{n}_*_principles_mean.npy"))
        except StopIteration:
            break
        pv = np.load(f_npy).astype(np.float32)            # (G, 784)
        rows.append((n, pv[:n_principles]))

    cols = n_principles
    sheet = Image.new("L", (cols * tile, len(rows) * tile), color=0)
    neurons = []
    for r, (n, pvs) in enumerate(rows):
        for c in range(cols):
            if c >= len(pvs):
                continue
            # layer-0 contribution vectors are small + signed; stretch each
            # tile by its own max |value| around mid-gray so the pattern shows.
            v = pvs[c].reshape(28, 28)
            scale = np.max(np.abs(v)) or 1.0
            img = np.clip(0.5 + 0.5 * v / scale, 0.0, 1.0)
            tile_img = Image.fromarray((img * 255).astype(np.uint8), "L")
            if tile != 28:
                tile_img = tile_img.resize((tile, tile), Image.NEAREST)
            sheet.paste(tile_img, (c * tile, r * tile))
        neurons.append({"neuron": n, "count": int(len(pvs))})

    out = WEB / "gallery" / name
    out.mkdir(parents=True, exist_ok=True)
    sheet.save(out / "sprite.png")
    write_json(out / "sprite.json",
               {"tile": tile, "cols": cols, "rows": len(rows), "neurons": neurons})
    print(f"  gallery {name}: {len(rows)}x{cols} tiles")


# --------------------------------------------------------------------------- #
def main():
    print("figures:")
    build_figures()
    print("models:")
    nets = {"best_mnist_deep_ln_drop": "mnist",
            "best_emnist_deep_ln_drop": "emnist"}
    for dirname, dataset in nets.items():
        d = MODELS / dirname
        build_model(d, dataset)
        build_gallery(d)
    print("done ->", WEB)


if __name__ == "__main__":
    main()
