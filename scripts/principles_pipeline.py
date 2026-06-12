"""Principle-vector pipeline with K-sweep dendrogram sharing.

For every model checkpoint under ``--root`` and every neuron in every layer:

  1. Forward-pass the training set and collect the per-neuron contribution
     vector ``V_ni = weight[ni] * layer_input`` for each training image.
  2. Take an N1-sized linspace subsample of those vectors and build ONE
     average-linkage cosine-distance dendrogram on the subsample. The
     cosine distance matrix is computed on the GPU as ``1 - X_hat @ X_hat.T``;
     the linkage call itself runs on the CPU (single-threaded, scipy).
  3. For every K in ``--k-values``: cut the dendrogram at K clusters,
     compute cluster centroids on the subsample, assign every full-set
     vector to its nearest centroid (cosine, GPU matmul), and write the
     principle vector for each cluster as the mean over all full-set
     vectors assigned to it.

The dendrogram is computed ONCE per neuron and reused for every K, so the
cost of K extra cluster counts is just K assignments + K means rather
than K full clusterings. This makes the K-sweep roughly an order of
magnitude faster than running the pipeline separately per K, because the
dominant cost (scipy hierarchical linkage at N1) is paid only once per
neuron.

Outputs land under each ``<config>/<seed>/principles_k<K>/<epoch>/`` in
exactly the layout the existing eval scripts expect, so this pipeline is
drop-in compatible with ``eval_principle_g7.py``, ``eval_subsets.py``,
``evaluate_principles.py``, etc.:

    <root>/<config>/<seed>/principles_k<K>/<epoch_stem>/
        layer{i}/
            neuron{j}_<in>in_<out>out_hierarchical_principles_mean.npy
            neuron{j}_<in>in_<out>out_hierarchical_groupings.npy   # only with --save-groupings
        meta.json

By default the per-sample cluster-label file ("groupings", shape (N,)) is
NOT written. It is only consumed by ad-hoc diagnostic tools (viewers,
group_histogram, etc.) and is ~882 KB per neuron per K at full EMNIST
size, which would dominate the disk footprint. Pass ``--save-groupings``
to emit it (downcast to int8/int16 to keep it compact).

Usage (from the repo root):

    # K-sweep on group_of_7 (epoch 200, all configs/seeds)
    python data_and_models_to_present/principles_pipeline.py group_of_7 \\
        --k-values 1,2,3,5,8,12,20,25,30,40,50,75,100 \\
        -n 200000 --dendrogram-samples 10000 \\
        --epochs 200 --device cuda --workers 4

    # Single K=5 on the subsets (all epochs)
    python data_and_models_to_present/principles_pipeline.py emnist_subsets \\
        --k-values 5 -n 200000 --dendrogram-samples 10000 \\
        --epochs 0 --device cuda --workers 4

Dependencies: numpy, scipy, torch, torchvision, safetensors.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Dataset loading (cached per process)
# ---------------------------------------------------------------------------

_DATASET_CACHE = {}


def _emnist_orient(img):
    # torchvision EMNIST ships rotated 90 CCW + horizontally mirrored vs
    # upright; the trainer applies this fix, so we replicate it here. Without
    # it the model sees garbage and the per-neuron contributions are
    # meaningless.
    return transforms.functional.hflip(transforms.functional.rotate(img, -90))


def _build_dataset(dataset_name, data_root):
    if dataset_name == "mnist":
        return datasets.MNIST(
            data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]),
        )
    if dataset_name == "emnist":
        return datasets.EMNIST(
            data_root, split="balanced", train=True, download=True,
            transform=transforms.Compose([
                transforms.Lambda(_emnist_orient),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]),
        )
    sys.exit(f"Unknown dataset: {dataset_name}")


def get_inputs(dataset_name, data_root, class_indices, num_samples):
    """Return (M, 784) float tensor of inputs from the requested class subset.

    ``class_indices`` may be ``None`` (use the entire training set) or a list
    of class ids (only those classes are eligible). ``M = min(num_samples,
    available_pool)``. Indices are linspaced across the eligible pool so the
    same rows are picked across runs.
    """
    key = (dataset_name, None if class_indices is None else tuple(class_indices),
           num_samples)
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    ds = _build_dataset(dataset_name, data_root)
    targets = ds.targets if hasattr(ds, "targets") else ds.train_labels
    targets = torch.as_tensor(targets)

    if class_indices is None:
        pool = torch.arange(len(ds))
    else:
        wanted = set(int(c) for c in class_indices)
        mask = torch.tensor([int(t) in wanted for t in targets], dtype=torch.bool)
        pool = mask.nonzero(as_tuple=True)[0]
    if pool.numel() == 0:
        sys.exit(f"No samples for class_indices={class_indices}")

    take = min(num_samples, int(pool.numel()))
    pick = pool[np.linspace(0, int(pool.numel()) - 1, take, dtype=np.int64)]
    images = torch.stack([ds[int(i)][0] for i in pick]).view(take, -1).float()
    _DATASET_CACHE[key] = images
    return images


# ---------------------------------------------------------------------------
# Model loading + forward pass
# ---------------------------------------------------------------------------

def load_model(path, device):
    """Load a safetensors trainer checkpoint as a list of layer dicts.

    Each entry is ``{"weight", "bias", optional "norm": {"weight","bias"}}``.
    The last entry is the classifier head (no norm, no activation). Layer
    discovery uses the trainer's key convention: ``linears.{i}.weight``,
    ``norms.{i}.weight``, ``head.weight``.
    """
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).to(device)

    hidden_idx = sorted(
        int(k.split(".")[1]) for k in tensors
        if k.startswith("linears.") and k.endswith(".weight")
    )
    layers = []
    for i in hidden_idx:
        layer = {
            "weight": tensors[f"linears.{i}.weight"],
            "bias":   tensors[f"linears.{i}.bias"],
        }
        if f"norms.{i}.weight" in tensors:
            layer["norm"] = {
                "weight": tensors[f"norms.{i}.weight"],
                "bias":   tensors[f"norms.{i}.bias"],
            }
        layers.append(layer)
    if "head.weight" in tensors:
        layers.append({
            "weight": tensors["head.weight"],
            "bias":   tensors["head.bias"],
        })
    if not layers:
        raise RuntimeError(f"No `linears.*` or `head` weights in {path}")
    return layers


def _layer_norm(x, ln, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return ln["weight"] * (x - mean) / torch.sqrt(var + eps) + ln["bias"]


@torch.no_grad()
def iter_layer_inputs(x, layers):
    """Yield ``(layer_idx, layer_input_tensor)`` in execution order."""
    for i, layer in enumerate(layers):
        yield i, x
        x = x @ layer["weight"].t() + layer["bias"]
        if "norm" in layer:
            x = _layer_norm(x, layer["norm"])
        if i != len(layers) - 1:
            x = torch.relu(x)


# ---------------------------------------------------------------------------
# Core algorithm: one dendrogram, many K values
# ---------------------------------------------------------------------------

@torch.no_grad()
def cluster_neuron_kshare(vectors_full, dendrogram_samples, k_values,
                          linkage_method, device, want_labels=False):
    """Cluster one neuron's contribution vectors at multiple K values.

    Builds one cosine-distance dendrogram from a linspace subsample of size
    ``dendrogram_samples`` and, for each ``K`` in ``k_values``:

      * cuts the dendrogram at K clusters
      * computes cluster centroids on the subsample
      * assigns every row of ``vectors_full`` to its nearest centroid (cosine)
      * emits the principle vector for each cluster as the mean over all
        full-set rows assigned to it

    ``vectors_full`` is expected on ``device`` already, shape ``(N, d)``.
    Returns ``{K: (labels_or_None, principles_float32_(K_actual, d))}``.
    ``labels`` is omitted (set to None) unless ``want_labels=True`` to avoid
    a (N,) array allocation per K when the caller only needs the principles.
    Labels are 1-indexed to match scipy's ``fcluster`` convention.
    """
    N, d = vectors_full.shape
    N1 = min(int(dendrogram_samples), N)

    # ---- linspace subsample
    sub_idx = torch.linspace(0, N - 1, N1, dtype=torch.long, device=device)
    sub = vectors_full.index_select(0, sub_idx)                       # (N1, d)

    # ---- cosine distance matrix on the subsample, on the GPU
    sub_n = sub / (sub.norm(dim=1, keepdim=True) + 1e-12)
    dist = 1.0 - sub_n @ sub_n.T
    dist.clamp_min_(0.0)                                              # floor float noise
    dist = 0.5 * (dist + dist.T)                                      # symmetrize
    dist.fill_diagonal_(0.0)

    # ---- linkage on the CPU (the per-neuron bottleneck; runs once)
    condensed = squareform(dist.cpu().numpy().astype(np.float64), checks=False)
    Z = linkage(condensed, method=linkage_method)

    # ---- L2-normalized full-set vectors stay on the GPU for fast assignment
    full_n = vectors_full / (vectors_full.norm(dim=1, keepdim=True) + 1e-12)

    out = {}
    for K in k_values:
        K_eff = max(1, min(int(K), N1))

        # cut dendrogram at K (1-indexed labels, range 1..K_actual)
        sub_labels_np = fcluster(Z, t=K_eff, criterion="maxclust").astype(np.int64)
        sub_labels = torch.from_numpy(sub_labels_np).to(device)

        # contiguous 0..K_actual-1 indexing for scatter ops
        present, inverse = torch.unique(sub_labels, return_inverse=True)
        K_actual = int(present.numel())

        # subsample centroids (normalized) -> nearest-centroid assignment
        sub_sums = torch.zeros(K_actual, d, device=device, dtype=sub_n.dtype)
        sub_counts = torch.zeros(K_actual, device=device, dtype=sub_n.dtype)
        sub_sums.index_add_(0, inverse, sub_n)
        sub_counts.index_add_(0, inverse,
                              torch.ones(N1, device=device, dtype=sub_n.dtype))
        centroids_n = sub_sums / sub_counts.unsqueeze(1).clamp_min_(1.0)
        centroids_n = centroids_n / (centroids_n.norm(dim=1, keepdim=True) + 1e-12)

        # assign full set: argmax cosine similarity -> local index in 0..K_actual-1
        assigned_local = (full_n @ centroids_n.T).argmax(dim=1)       # (N,)

        # final principles: mean over full-set vectors per cluster
        sums = torch.zeros(K_actual, d, device=device, dtype=vectors_full.dtype)
        counts = torch.zeros(K_actual, device=device, dtype=vectors_full.dtype)
        sums.index_add_(0, assigned_local, vectors_full)
        counts.index_add_(0, assigned_local,
                          torch.ones(N, device=device, dtype=vectors_full.dtype))
        principles = sums / counts.unsqueeze(1).clamp_min_(1.0)

        if want_labels:
            labels_full = present[assigned_local].cpu().numpy().astype(np.int64)
        else:
            labels_full = None
        out[int(K)] = (labels_full, principles.cpu().numpy().astype(np.float32))

    return out


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------

def process_model(payload):
    cfg            = payload["cfg"]
    seed_dir       = Path(payload["seed_dir"])
    ckpt_path      = Path(payload["ckpt_path"])
    data_root      = payload["data_root"]
    num_samples    = payload["num_samples"]
    dendro_samples = payload["dendrogram_samples"]
    k_values       = payload["k_values"]
    linkage_method = payload["linkage_method"]
    device_str     = payload["device"]
    skip_existing  = payload["skip_existing"]
    save_groupings = payload.get("save_groupings", False)

    device = torch.device(device_str)
    epoch_stem = ckpt_path.stem
    out_roots = {K: seed_dir / f"principles_k{K}" / epoch_stem for K in k_values}

    if skip_existing and all((p / "meta.json").exists() for p in out_roots.values()):
        return f"[skip] {ckpt_path}"
    for p in out_roots.values():
        p.mkdir(parents=True, exist_ok=True)

    class_indices = cfg.get("class_indices")  # None for plain mnist/emnist
    inputs = get_inputs(cfg["dataset"], data_root, class_indices, num_samples).to(device)
    layers = load_model(str(ckpt_path), device)

    n_neurons = 0
    layer_shapes = []
    for li, layer_input in iter_layer_inputs(inputs, layers):
        weight = layers[li]["weight"]                                  # (out, in)
        out_size, in_size = weight.shape
        layer_shapes.append([int(out_size), int(layer_input.shape[0]), int(in_size)])

        layer_dirs = {K: out_roots[K] / f"layer{li}" for K in k_values}
        for d in layer_dirs.values():
            d.mkdir(exist_ok=True)

        for ni in range(out_size):
            # per-neuron contribution vectors: (N, in)
            vectors = weight[ni].unsqueeze(0) * layer_input
            results = cluster_neuron_kshare(
                vectors, dendro_samples, k_values, linkage_method, device,
                want_labels=save_groupings)

            base_name = f"neuron{ni}_{in_size}in_{out_size}out"
            for K, (labels, principles) in results.items():
                base = layer_dirs[K] / base_name
                # Filenames mirror the existing pipeline so eval scripts and
                # `PrincipleLayer.from_folder` find these without changes.
                np.save(str(base) + "_hierarchical_principles_mean.npy",
                        principles)
                if save_groupings and labels is not None:
                    # Per-sample cluster labels are 1..K_actual; downcast to
                    # the smallest signed type that fits (int8 covers K<=127,
                    # which is the entire default K sweep) so the (N,) array
                    # stays ~110 KB instead of ~880 KB at full EMNIST size.
                    lmax = int(labels.max()) if labels.size else 0
                    if lmax < 128:
                        dt = np.int8
                    elif lmax < 32768:
                        dt = np.int16
                    else:
                        dt = np.int32
                    np.save(str(base) + "_hierarchical_groupings.npy",
                            labels.astype(dt, copy=False))
            n_neurons += 1

    meta_base = {
        "checkpoint": str(ckpt_path),
        "config": cfg,
        "num_samples": int(num_samples),
        "dendrogram_samples": int(dendro_samples),
        "cluster_method": "hierarchical",
        "linkage_method": linkage_method,
        "metric": "cosine",
        "pv_method": "mean",
        "save_groupings": bool(save_groupings),
        "n_layers": len(layer_shapes),
        "layer_shapes": layer_shapes,
        "n_neurons": n_neurons,
    }
    for K in k_values:
        meta = dict(meta_base, num_clusters=int(K))
        with open(out_roots[K] / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return (f"[ok]   {ckpt_path}: {n_neurons} neurons x {len(k_values)} K "
            f"= {n_neurons * len(k_values)} principle vector sets")


# ---------------------------------------------------------------------------
# Discovery + CLI
# ---------------------------------------------------------------------------

def discover_models(root, epoch_filter):
    """Walk ``root/<config>/<seed>/epoch*.safetensors`` and yield jobs."""
    root = Path(root)
    for cfg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in cfg_dir.iterdir() if p.is_dir()):
            cfg_path = seed_dir / "config.json"
            if not cfg_path.exists():
                continue
            with open(cfg_path) as f:
                cfg = json.load(f)
            for ckpt in sorted(seed_dir.glob("epoch*.safetensors")):
                try:
                    ep = int(ckpt.stem.replace("epoch", ""))
                except ValueError:
                    continue
                if epoch_filter and ep not in epoch_filter:
                    continue
                yield cfg, seed_dir, ckpt


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("root",
                   help="Folder containing per-config model dirs "
                        "(group_of_7, emnist_subsets, deep_emnist_subsets, "
                        "wider_deeper).")
    p.add_argument("--k-values", default="1,2,3,5,8,12,20,25,30,40,50,75,100",
                   help="Comma-separated K values to cut the dendrogram at "
                        "(default: 1,2,3,5,8,12,20,25,30,40,50,75,100).")
    p.add_argument("--epochs", type=int, nargs="+", default=[200],
                   help="Which epoch checkpoints to process (default: 200; "
                        "pass 0 for ALL epochs).")
    p.add_argument("--data-root", default="./data",
                   help="Where torchvision datasets live (default: ./data).")
    p.add_argument("-n", "--num-samples", type=int, default=200_000,
                   help="Full-set sample cap (default: 200000; capped to "
                        "dataset size, so this means 'use the whole training "
                        "pool' for both MNIST and EMNIST).")
    p.add_argument("--dendrogram-samples", type=int, default=10_000,
                   help="Subsample size used to build the dendrogram "
                        "(default: 10000). The full --num-samples set is "
                        "still used to compute the principle vectors as means "
                        "over the full assigned membership.")
    p.add_argument("--linkage-method", default="average",
                   choices=["single", "complete", "average", "weighted"],
                   help="scipy linkage method (default: average -- matches "
                        "the existing `cluster_hierarchical` implementation).")
    p.add_argument("--device", default="cpu",
                   help="torch device for distance matrix + assignment "
                        "(default: cpu; 'cuda' is dramatically faster).")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel model workers (default: 1; 0 = cpu_count). "
                        "Each worker uses ~1-2 GB VRAM at the recommended "
                        "settings; 4 fits comfortably in 10 GB.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip checkpoints whose every K output dir already "
                        "has meta.json.")
    p.add_argument("--save-groupings", action="store_true",
                   help="Also write the per-sample cluster-label file "
                        "(neuron*_hierarchical_groupings.npy). Off by "
                        "default; eval scripts and PrincipleLayer don't "
                        "need it. Adds ~110 KB per neuron per K (int8) at "
                        "full EMNIST size -- only enable if you need it for "
                        "ad-hoc viewers / histograms.")
    args = p.parse_args()

    try:
        k_values = sorted({int(x) for x in args.k_values.split(",") if x.strip()})
    except ValueError:
        sys.exit(f"Invalid --k-values: {args.k_values!r}")
    if not k_values:
        sys.exit("--k-values must list at least one integer")

    epoch_filter = None if args.epochs == [0] else set(args.epochs)
    jobs = list(discover_models(args.root, epoch_filter))
    if not jobs:
        sys.exit(f"No matching checkpoints found under {args.root}")
    print(f"Discovered {len(jobs)} checkpoints; K values: {k_values}")

    workers = args.workers if args.workers != 0 else (os.cpu_count() or 1)
    workers = max(1, min(workers, len(jobs)))

    payloads = [
        {
            "cfg": cfg,
            "seed_dir": str(seed_dir),
            "ckpt_path": str(ckpt),
            "data_root": args.data_root,
            "num_samples": args.num_samples,
            "dendrogram_samples": args.dendrogram_samples,
            "k_values": k_values,
            "linkage_method": args.linkage_method,
            "device": args.device,
            "skip_existing": args.skip_existing,
            "save_groupings": args.save_groupings,
        }
        for (cfg, seed_dir, ckpt) in jobs
    ]

    t0 = time.time()
    ok = fail = 0
    if workers == 1:
        for pl in payloads:
            try:
                print(process_model(pl))
                ok += 1
            except Exception as e:
                print(f"[fail] {pl['ckpt_path']}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                fail += 1
    else:
        print(f"Running {len(payloads)} jobs across {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_model, pl): pl for pl in payloads}
            for fut in as_completed(futures):
                pl = futures[fut]
                try:
                    print(fut.result())
                    ok += 1
                except Exception as e:
                    print(f"[fail] {pl['ckpt_path']}: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    fail += 1

    print(f"\nDone in {time.time() - t0:.1f}s. ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
