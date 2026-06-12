"""Generate base + principal accuracy data for the newly-kshare-trained
group_of_7 / emnist_subsets / deep_emnist_subsets model trees.

For each tree, discover all `principles_kN/` subdirs present under any seed
dir, then for each K run eval_subsets.evaluate_one() against every checkpoint
in `--epochs`, in both `replace=none` (base NN) and `replace=hidden`
(principle layers substituted) modes. Writes one CSV per (tree, K) under this
directory.

The dataset filter cache inside eval_subsets is module-level, so all K runs
share it (the EMNIST/MNIST tensors load once per (dataset, class_indices)).
"""
import csv
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import eval_subsets as es  # noqa: E402

ROOTS = [
    ("emnist_subsets",      ROOT / "emnist_subsets"),
    ("deep_emnist_subsets", ROOT / "deep_emnist_subsets"),
    ("group_of_7",          ROOT / "group_of_7"),
]
EPOCHS = {10, 25, 50, 75, 100, 150, 200}
DATA_ROOT = str(ROOT / "data")
BATCH_SIZE = 1024
METHOD = "mean"
DTYPE = "float32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")

OUT_COLS = ["dataset", "arch", "config", "subset", "subset_size",
            "seed", "epoch", "replace", "K", "train_acc", "test_acc", "ckpt"]


def discover_ks(root: Path):
    ks = set()
    if not root.is_dir():
        return []
    for cfg_dir in root.iterdir():
        if not cfg_dir.is_dir():
            continue
        for seed_dir in cfg_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            for sub in seed_dir.iterdir():
                if sub.is_dir() and sub.name.startswith("principles_k"):
                    try:
                        ks.add(int(sub.name[len("principles_k"):]))
                    except ValueError:
                        pass
    return sorted(ks)


def ensure_class_indices(cfg):
    """group_of_7 configs are trained on the full dataset and omit
    class_indices/subset_name. Synthesize them so eval_subsets can filter
    (a no-op filter) and remap (identity)."""
    if "class_indices" not in cfg:
        cfg = dict(cfg)
        cfg["class_indices"] = list(range(int(cfg["num_classes"])))
        cfg.setdefault("subset_name", "")
    return cfg


def run_tree(name: str, root: Path):
    ks = discover_ks(root)
    print(f"\n=== {name}: K values = {ks} ===")
    if not ks:
        print(f"  (no principles_k* subdirs found under {root}; skipping)")
        return

    # All (cfg, seed_dir, ckpt, epoch_stem) for this tree, filtered to EPOCHS.
    all_jobs = list(es.discover(str(root), EPOCHS))
    all_jobs = [(ensure_class_indices(cfg), sd, ck, ep)
                for (cfg, sd, ck, ep) in all_jobs]
    print(f"  total checkpoints (all K): {len(all_jobs)}")

    for K in ks:
        subdir = f"principles_k{K}"
        out_csv = HERE / f"{name}_k{K}.csv"
        if out_csv.exists():
            print(f"  [skip] {out_csv.name} already exists")
            continue

        jobs = [(cfg, sd, ck, ep) for (cfg, sd, ck, ep) in all_jobs
                if (sd / subdir).is_dir()]
        if not jobs:
            print(f"  K={K}: no seed dirs have {subdir}/ — skipping")
            continue

        print(f"  K={K}: {len(jobs)} ckpts x 2 modes = {len(jobs)*2} jobs")
        rows = []
        t0 = time.time()
        for i, (cfg, sd, ck, ep) in enumerate(jobs, 1):
            for mode in ("none", "hidden"):
                payload = {
                    "cfg": cfg,
                    "ckpt_path": str(ck),
                    "seed_dir": str(sd),
                    "epoch_stem": ep,
                    "data_root": DATA_ROOT,
                    "batch_size": BATCH_SIZE,
                    "method": METHOD,
                    "replace_mode": mode,
                    "dtype": DTYPE,
                    "device": DEVICE,
                    "principles_subdir": subdir,
                }
                try:
                    r = es.evaluate_one(payload)
                    r["K"] = K
                    rows.append(r)
                except Exception as e:
                    print(f"  [fail] {ck} K={K} mode={mode}: "
                          f"{type(e).__name__}: {e}")
            if i % 25 == 0 or i == len(jobs):
                dt = time.time() - t0
                rate = i / dt if dt else 0
                eta = (len(jobs) - i) / rate if rate else 0
                print(f"    [{i:>4}/{len(jobs)}] elapsed {dt:>6.0f}s  "
                      f"rate {rate:.2f} ckpt/s  ETA {eta:>5.0f}s")

        tmp = out_csv.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUT_COLS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in OUT_COLS})
        tmp.replace(out_csv)
        print(f"  wrote {out_csv.name} ({len(rows)} rows) in "
              f"{time.time()-t0:.0f}s")


def main():
    t_all = time.time()
    for name, root in ROOTS:
        run_tree(name, root)
    print(f"\n[done] total wall time {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
