"""Validate the browser inference binaries by reimplementing the exact
forward pass (linear + principal) in numpy, reading only the emitted
web/models/<net>/{meta.json,linear.f32,principal.i8} — i.e. the same bytes
the JS engine consumes. Reports test accuracy on real MNIST/EMNIST images.

This doubles as the reference spec for pv-infer.js.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEB = ROOT / "web"
DATA_ROOT = ROOT / "raw_data" / "_torchdata"


def load_net(name):
    d = WEB / "models" / name
    meta = json.loads((d / "meta.json").read_text())
    f32 = np.frombuffer((d / "linear.f32").read_bytes(), dtype=np.float32)
    i8 = np.frombuffer((d / "principal.i8").read_bytes(), dtype=np.int8)
    return meta, f32, i8


def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(); var = x.var()
    return g * (x - mean) / np.sqrt(var + eps) + b


def forward_linear(meta, f32, x):
    for ly in meta["layers"]:
        oi, ii = ly["out"], ly["in"]
        w = f32[ly["wOff"] // 4: ly["wOff"] // 4 + oi * ii].reshape(oi, ii)
        b = f32[ly["bOff"] // 4: ly["bOff"] // 4 + oi]
        x = w @ x + b
        if "ngOff" in ly:
            g = f32[ly["ngOff"] // 4: ly["ngOff"] // 4 + oi]
            bn = f32[ly["nbOff"] // 4: ly["nbOff"] // 4 + oi]
            x = layer_norm(x, g, bn)
        if ly["relu"]:
            x = np.maximum(x, 0.0)
    return x


def forward_principal(meta, f32, i8, x):
    for ly in meta["layers"]:
        oi, ii = ly["out"], ly["in"]
        w = f32[ly["wOff"] // 4: ly["wOff"] // 4 + oi * ii].reshape(oi, ii)
        b = f32[ly["bOff"] // 4: ly["bOff"] // 4 + oi]
        g = ly["g"]
        rs = f32[ly["rsOff"] // 4: ly["rsOff"] // 4 + sum(g)]
        pre = np.empty(oi, dtype=np.float32)
        pv_cursor = ly["pvOff"]
        rs_cursor = 0
        for n in range(oi):
            gc = g[n]
            block = i8[pv_cursor: pv_cursor + gc * ii].reshape(gc, ii).astype(np.float32)
            pv_cursor += gc * ii
            h = w[n] * x                       # Hadamard contribution
            scores = block @ h                 # argmax cosine (|h| const per neuron)
            gstar = int(np.argmax(scores))
            pre[n] = rs[rs_cursor + gstar] + b[n]
            rs_cursor += gc
        x = pre
        if "ngOff" in ly:
            gn = f32[ly["ngOff"] // 4: ly["ngOff"] // 4 + oi]
            bn = f32[ly["nbOff"] // 4: ly["nbOff"] // 4 + oi]
            x = layer_norm(x, gn, bn)
        if ly["relu"]:
            x = np.maximum(x, 0.0)
    return x


def _emnist_orient(img):
    return img.rotate(-90, expand=True).transpose(0)  # PIL Image.FLIP_LEFT_RIGHT == 0


def get_test(dataset, n):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    if dataset == "mnist":
        ds = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=tf)
    else:
        tf = transforms.Compose([transforms.Lambda(_emnist_orient),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
        ds = datasets.EMNIST(str(DATA_ROOT), split="balanced", train=False,
                             download=True, transform=tf)
    idx = np.linspace(0, len(ds) - 1, n, dtype=int)
    xs = np.stack([ds[int(i)][0].view(-1).numpy() for i in idx])
    ys = np.array([int(ds[int(i)][1]) for i in idx])
    return xs, ys


def run(name, dataset, n=300):
    meta, f32, i8 = load_net(name)
    xs, ys = get_test(dataset, n)
    lin_ok = prin_ok = 0
    for x, y in zip(xs, ys):
        if int(np.argmax(forward_linear(meta, f32, x))) == y:
            lin_ok += 1
        if int(np.argmax(forward_principal(meta, f32, i8, x))) == y:
            prin_ok += 1
    print(f"{name:28s} n={n}  linear={lin_ok/n:6.3f}  principal={prin_ok/n:6.3f}")


if __name__ == "__main__":
    run("best_mnist_deep_ln_drop", "mnist")
    run("best_emnist_deep_ln_drop", "emnist")
