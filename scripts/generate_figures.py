"""Generate figures + CSV datasets for the captions in
``Closer to final draft - prin.md`` (the "deeper/wider" Figure 5 is omitted
on request).

Outputs are written next to this script:

    figure1_louvain_accuracy.{csv,png}
    figure2_shallow_mnist_neuron_substitution.{csv,png}
    figure3_shallow_emnist_label_subsets.{csv,png}
    figure4_deep_emnist_label_subsets.{csv,png}
    figure6_per_layer_substitution.{csv,png}
    figure7_training_epochs.{csv,png}
    figure8_groups_1_to_20.{csv,png}
    figure9_groups_beyond_20.{csv,png}
    figure10_emnist_ln_k100_epochs.{csv,png}

Conventions:
    * principal model baseline is k = 20 (Louvain typically lands in 15-25),
      unless the caption asks for a different k.
    * Linear ("none") and principal ("hidden") rows for the same
      (dataset, arch, config, seed, epoch) are matched and averaged across
      seeds; error bars show one stddev.
    * epoch200 is used as the "trained model" snapshot.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- paths --------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACC = ROOT / "accuracy_data"

K_BASELINE = 20  # principal-model baseline number of groups (proxy for Louvain)
MAIN_EPOCH = "epoch200"
SUBSET_SIZES = {
    "ten_1": 10, "fifteen_1": 15, "twenty_1": 20, "twenty_five_1": 25,
    "thirty_1": 30, "thirty_five_1": 35, "fourty_1": 40,
}
CONFIG_LABEL = {
    "plain": "plain", "drop": "dropout", "ln": "layernorm",
    "ln_drop": "layernorm+dropout",
}
EPOCH_INT = lambda e: int(re.sub(r"\D", "", e))


# -- helpers ------------------------------------------------------------------
def read_g7(k: int) -> pd.DataFrame:
    df = pd.read_csv(ACC / f"group_of_7_k{k}.csv")
    df["epoch_n"] = df["epoch"].map(EPOCH_INT)
    return df


def matched(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot replace=none vs hidden into one row per (dataset, arch, config,
    seed, epoch_n, subset, K)."""
    keys = ["dataset", "arch", "config", "subset", "subset_size",
            "seed", "epoch", "epoch_n", "K"]
    df = df.copy()
    df["subset"] = df["subset"].fillna("__none__")
    pieces = []
    for replace in ("none", "hidden"):
        sub = df[df["replace"] == replace][keys + ["test_acc", "train_acc"]]
        sub = sub.rename(columns={
            "test_acc": f"test_acc_{replace}",
            "train_acc": f"train_acc_{replace}",
        })
        pieces.append(sub)
    m = pieces[0].merge(pieces[1], on=keys, how="inner")
    return m


def aggregate(df: pd.DataFrame, group_cols, value_cols):
    g = df.groupby(group_cols, dropna=False)
    out = g[value_cols].agg(["mean", "std", "count"])
    out.columns = [f"{v}_{stat}" for v, stat in out.columns]
    return out.reset_index()


def save_plot(fig, name: str):
    fig.tight_layout()
    fig.savefig(HERE / f"{name}.png", dpi=140)
    plt.close(fig)


# -- figure 1: Louvain test accuracy (MNIST + EMNIST) -------------------------
def figure1():
    df = read_g7(K_BASELINE)
    df = df[df.epoch == MAIN_EPOCH]
    m = matched(df)
    agg = aggregate(
        m, ["dataset", "arch", "config"],
        ["test_acc_none", "test_acc_hidden"],
    )
    agg.to_csv(HERE / "figure1_louvain_accuracy.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, dataset in zip(axes, ("mnist", "emnist")):
        sub = agg[agg.dataset == dataset].copy()
        configs = ["plain", "drop", "ln", "ln_drop"]
        archs = ["shallow", "deep"]
        x = np.arange(len(configs))
        width = 0.18
        offsets = [-1.5, -0.5, 0.5, 1.5]
        bars = []
        for off, (arch, replace) in zip(
            offsets,
            [("shallow", "none"), ("shallow", "hidden"),
             ("deep", "none"), ("deep", "hidden")],
        ):
            ys, errs = [], []
            for cfg in configs:
                row = sub[(sub.arch == arch) & (sub.config == cfg)]
                if row.empty:
                    ys.append(np.nan); errs.append(0)
                else:
                    ys.append(row[f"test_acc_{replace}_mean"].iloc[0])
                    errs.append(row[f"test_acc_{replace}_std"].iloc[0])
            label = f"{arch} {'linear' if replace=='none' else 'principal'}"
            bars.append(ax.bar(x + off * width, ys, width, yerr=errs,
                                capsize=3, label=label))
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABEL[c] for c in configs], rotation=15)
        ax.set_title(f"{dataset.upper()} (k={K_BASELINE} principal baseline)")
        ax.set_ylabel("test accuracy")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Figure 1 - test accuracy with Louvain-style grouping",
                 y=1.02)
    save_plot(fig, "figure1_louvain_accuracy")


# -- figure 2: shallow MNIST per-neuron substitution --------------------------
def figure2():
    src = ROOT / "sweep_hybrid_out" / "sweep_hybrid.csv"
    df = pd.read_csv(src)
    # Average across seeds + orderings for each (layer, k).
    agg = (df.groupby(["layer", "ordering", "k"], as_index=False)
              .agg(acc_mean=("accuracy", "mean"),
                   acc_std=("accuracy", "std"),
                   n=("accuracy", "count")))
    # Normalize each (layer, ordering) curve to k=0 baseline.
    base = (agg[agg.k == 0]
            .set_index(["layer", "ordering"])["acc_mean"].to_dict())
    agg["acc_norm_mean"] = agg.apply(
        lambda r: r.acc_mean / base[(r.layer, r.ordering)], axis=1)
    agg["acc_norm_std"] = agg.apply(
        lambda r: r.acc_std / base[(r.layer, r.ordering)]
        if r.acc_std == r.acc_std else 0.0, axis=1)
    agg.to_csv(HERE / "figure2_shallow_mnist_neuron_substitution.csv",
               index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"fc1": "tab:blue", "fc2": "tab:orange"}
    styles = {"index": "-", "random": "--"}
    for (layer, ordering), grp in agg.groupby(["layer", "ordering"]):
        grp = grp.sort_values("k")
        ax.plot(grp.k, grp.acc_norm_mean,
                color=colors[layer], linestyle=styles[ordering],
                label=f"{layer} ({ordering})")
        ax.fill_between(grp.k,
                         grp.acc_norm_mean - grp.acc_norm_std,
                         grp.acc_norm_mean + grp.acc_norm_std,
                         color=colors[layer], alpha=0.12)
    ax.set_xlabel("# principal neurons substituted in (of 128)")
    ax.set_ylabel("test accuracy (normalized to k=0)")
    ax.set_title("Figure 2 - shallow MNIST per-neuron principal substitution")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
    save_plot(fig, "figure2_shallow_mnist_neuron_substitution")


# -- figures 3 & 4: subset-size label sweeps ----------------------------------
def _subset_figure(csv_name: str, depth_label: str, fig_index: int,
                    out_stem: str):
    raw = pd.read_csv(ACC / csv_name)
    raw["epoch_n"] = raw["epoch"].map(EPOCH_INT)
    raw = raw[raw.epoch == MAIN_EPOCH]
    raw["subset_size"] = raw["subset"].map(SUBSET_SIZES).fillna(raw["subset_size"])

    m = matched(raw)
    m["delta"] = m.test_acc_none - m.test_acc_hidden
    agg = aggregate(
        m, ["config", "subset_size"],
        ["test_acc_none", "test_acc_hidden", "delta"],
    ).sort_values(["config", "subset_size"])
    agg.to_csv(HERE / f"{out_stem}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    cmap = plt.get_cmap("tab10")
    for i, cfg in enumerate(sorted(agg.config.unique())):
        sub = agg[agg.config == cfg].sort_values("subset_size")
        ax.errorbar(sub.subset_size, sub.delta_mean, yerr=sub.delta_std,
                    marker="o", capsize=3, color=cmap(i),
                    label=CONFIG_LABEL.get(cfg, cfg))
    ax.set_xlabel("# EMNIST-balanced classes in subset")
    ax.set_ylabel("linear − principal test accuracy")
    ax.set_title(f"Figure {fig_index} - {depth_label} EMNIST subset sweep "
                 f"(k={K_BASELINE})")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
    save_plot(fig, out_stem)


def figure3():
    _subset_figure("emnist_subsets_k20.csv", "shallow", 3,
                   "figure3_shallow_emnist_label_subsets")


def figure4():
    _subset_figure("deep_emnist_subsets_k20.csv", "deep", 4,
                   "figure4_deep_emnist_label_subsets")


# -- figure 6: per-layer single / prefix --------------------------------------
def figure6():
    src = ROOT / "eval_g7_per_layer_epoch200.csv"
    df = pd.read_csv(src)
    df = df[df.missing_principles == False]

    def parse(variant):
        if variant == "baseline":
            return "baseline", -1
        m = re.match(r"(single|prefix)-(\d+)", variant)
        return m.group(1), int(m.group(2))

    df[["kind", "layer"]] = df.variant.apply(
        lambda v: pd.Series(parse(v)))

    agg = (df.groupby(["dataset", "arch", "config", "kind", "layer"],
                      as_index=False)
              .agg(test_mean=("test_acc", "mean"),
                   test_std=("test_acc", "std"),
                   n=("test_acc", "count")))
    agg.to_csv(HERE / "figure6_per_layer_substitution.csv", index=False)

    pairs = [("mnist", "shallow"), ("mnist", "deep"),
             ("emnist", "shallow"), ("emnist", "deep")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=False)
    for ax, (dataset, arch) in zip(axes.flat, pairs):
        sub = agg[(agg.dataset == dataset) & (agg.arch == arch)]
        if sub.empty:
            ax.set_visible(False); continue
        configs = sorted(sub.config.unique())
        for i, cfg in enumerate(configs):
            color = plt.get_cmap("tab10")(i)
            ss = sub[sub.config == cfg]
            base = ss[ss.kind == "baseline"]
            if not base.empty:
                ax.axhline(base.test_mean.iloc[0], color=color,
                           linestyle=":", alpha=0.7,
                           label=f"{CONFIG_LABEL.get(cfg, cfg)} baseline")
            for kind, marker in (("single", "o"), ("prefix", "s")):
                kk = ss[ss.kind == kind].sort_values("layer")
                if kk.empty:
                    continue
                ax.errorbar(kk.layer, kk.test_mean, yerr=kk.test_std,
                            marker=marker, capsize=2, color=color,
                            linestyle="-" if kind == "single" else "--",
                            label=f"{CONFIG_LABEL.get(cfg, cfg)} {kind}")
        ax.set_title(f"{dataset.upper()} {arch}")
        ax.set_xlabel("layer index")
        ax.set_ylabel("test accuracy")
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle(
        f"Figure 6 - single vs prefix principal-layer substitution "
        f"(k={K_BASELINE})", y=1.0)
    save_plot(fig, "figure6_per_layer_substitution")


# -- figure 7: model performance across training epochs ----------------------
def figure7():
    df = read_g7(K_BASELINE)
    m = matched(df)
    agg = aggregate(
        m, ["dataset", "arch", "config", "epoch_n"],
        ["test_acc_none", "test_acc_hidden"],
    ).sort_values(["dataset", "arch", "config", "epoch_n"])
    agg.to_csv(HERE / "figure7_training_epochs.csv", index=False)

    pairs = [("mnist", "shallow"), ("mnist", "deep"),
             ("emnist", "shallow"), ("emnist", "deep")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (dataset, arch) in zip(axes.flat, pairs):
        sub = agg[(agg.dataset == dataset) & (agg.arch == arch)]
        for i, cfg in enumerate(sorted(sub.config.unique())):
            color = plt.get_cmap("tab10")(i)
            ss = sub[sub.config == cfg].sort_values("epoch_n")
            ax.errorbar(ss.epoch_n, ss.test_acc_none_mean,
                        yerr=ss.test_acc_none_std, marker="o", capsize=2,
                        color=color, linestyle="-",
                        label=f"{CONFIG_LABEL.get(cfg, cfg)} linear")
            ax.errorbar(ss.epoch_n, ss.test_acc_hidden_mean,
                        yerr=ss.test_acc_hidden_std, marker="s", capsize=2,
                        color=color, linestyle="--",
                        label=f"{CONFIG_LABEL.get(cfg, cfg)} principal")
        ax.set_title(f"{dataset.upper()} {arch}")
        ax.set_xlabel("training epoch")
        ax.set_ylabel("test accuracy")
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f"Figure 7 - test accuracy over training (k={K_BASELINE})",
                 y=1.0)
    save_plot(fig, "figure7_training_epochs")


# -- figures 8 & 9: group-count sweep -----------------------------------------
def _groups_sweep(ks, fig_index, out_stem, dataset_filter=None, title=None):
    frames = []
    for k in ks:
        df = read_g7(k)
        df = df[df.epoch == MAIN_EPOCH]
        if dataset_filter is not None:
            df = df[df.dataset.isin(dataset_filter)]
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    m = matched(big)
    # Principal model performance per K.
    agg = (m.groupby(["dataset", "arch", "config", "K"], as_index=False)
              .agg(prin_mean=("test_acc_hidden", "mean"),
                   prin_std=("test_acc_hidden", "std"),
                   linear_mean=("test_acc_none", "mean"),
                   linear_std=("test_acc_none", "std"),
                   n=("test_acc_hidden", "count"))
              .sort_values(["dataset", "arch", "config", "K"]))
    agg.to_csv(HERE / f"{out_stem}.csv", index=False)

    datasets = sorted(agg.dataset.unique())
    archs = ["shallow", "deep"]
    fig, axes = plt.subplots(len(datasets), len(archs),
                              figsize=(4.8 * len(archs), 3.6 * len(datasets)),
                              squeeze=False, sharex=True)
    for r, dataset in enumerate(datasets):
        for c, arch in enumerate(archs):
            ax = axes[r, c]
            sub = agg[(agg.dataset == dataset) & (agg.arch == arch)]
            if sub.empty:
                ax.set_visible(False); continue
            for i, cfg in enumerate(sorted(sub.config.unique())):
                ss = sub[sub.config == cfg].sort_values("K")
                color = plt.get_cmap("tab10")(i)
                ax.errorbar(ss.K, ss.prin_mean, yerr=ss.prin_std,
                            marker="o", capsize=2, color=color,
                            label=CONFIG_LABEL.get(cfg, cfg))
                lin = ss.linear_mean.mean()
                ax.axhline(lin, color=color, linestyle=":", alpha=0.5)
            ax.set_title(f"{dataset.upper()} {arch}")
            ax.set_xlabel("# groups (K)")
            ax.set_ylabel("test accuracy")
            ax.grid(linestyle=":", alpha=0.5)
            ax.legend(fontsize=6)
    fig.suptitle(title or f"Figure {fig_index} - group count sweep", y=1.0)
    save_plot(fig, out_stem)


def figure8():
    _groups_sweep([1, 2, 3, 5, 8, 12, 20], 8, "figure8_groups_1_to_20",
                  title="Figure 8 - principal model performance, K=1..20")


def figure9():
    _groups_sweep([20, 25, 30, 40, 50, 75, 100], 9,
                  "figure9_groups_beyond_20",
                  dataset_filter=["emnist"],
                  title=("Figure 9 - principal model performance, "
                         "K=20..100 (EMNIST)"))


# -- figure 10: EMNIST LN k=100 over epochs ----------------------------------
def figure10():
    df = read_g7(100)
    df = df[(df.dataset == "emnist") & (df.config == "ln")]
    m = matched(df)
    agg = aggregate(
        m, ["arch", "epoch_n"],
        ["test_acc_none", "test_acc_hidden"],
    ).sort_values(["arch", "epoch_n"])
    agg.to_csv(HERE / "figure10_emnist_ln_k100_epochs.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for i, arch in enumerate(("shallow", "deep")):
        sub = agg[agg.arch == arch].sort_values("epoch_n")
        color = plt.get_cmap("tab10")(i)
        ax.errorbar(sub.epoch_n, sub.test_acc_none_mean,
                    yerr=sub.test_acc_none_std,
                    marker="o", capsize=2, color=color, linestyle="-",
                    label=f"{arch} linear")
        ax.errorbar(sub.epoch_n, sub.test_acc_hidden_mean,
                    yerr=sub.test_acc_hidden_std,
                    marker="s", capsize=2, color=color, linestyle="--",
                    label=f"{arch} principal (k=100)")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("test accuracy")
    ax.set_title("Figure 10 - EMNIST layernorm models over training")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
    save_plot(fig, "figure10_emnist_ln_k100_epochs")


# -- driver -------------------------------------------------------------------
FIGS = [
    ("figure1", figure1),
    ("figure2", figure2),
    ("figure3", figure3),
    ("figure4", figure4),
    ("figure6", figure6),
    ("figure7", figure7),
    ("figure8", figure8),
    ("figure9", figure9),
    ("figure10", figure10),
]


def main():
    for name, fn in FIGS:
        print(f"== {name} ==")
        fn()
    print("done")


if __name__ == "__main__":
    main()
