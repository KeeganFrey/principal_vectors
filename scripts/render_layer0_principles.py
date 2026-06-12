"""Render layer-0 principle vectors directly as 28x28 PNG images.

The first hidden layer (fc1) takes the raw MNIST image as input, so each of
its principle vectors is already a 784-dim point in image space. "Rendering
backwards" reduces to: reshape -> un-normalize the MNIST transform ->
rescale to [0, 255]. No iterative inversion is required (use
optimize_principle_input.py for layers deeper than the first).

Inputs accepted:
  - A single .npy file (shape (G, 784)): renders one PNG per row.
  - A folder of *_principles_<method>.npy files (e.g. augmented/layer0_in):
    renders one subfolder per neuron, one PNG per principle.

Two rescale modes:
  --mode mnist    treat the vector as MNIST-normalized pixels and undo the
                  transform (x * 0.3081 + 0.1307), then clip to [0, 1].
                  Best when the principle vector was fit on
                  Normalize(0.1307, 0.3081) inputs.
  --mode signed   center 0 -> mid-gray, scale by max(|x|) so the sign of the
                  direction is visible. Best for raw direction vectors.

Examples
--------
    # One neuron's principle vectors, MNIST un-normalize:
    python render_layer0_principles.py \
        augmented/layer0_in/layer0_neuron0_784in_128out_louvain_principles_mean.npy \
        -o layer0_neuron0_png

    # Every neuron in the folder, signed colormap:
    python render_layer0_principles.py augmented/layer0_in \
        -o layer0_principles_png --mode signed
"""
import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def rescale_mnist(vec: np.ndarray) -> np.ndarray:
    img = vec * MNIST_STD + MNIST_MEAN
    return np.clip(img, 0.0, 1.0)


def rescale_signed(vec: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(vec)))
    if m < 1e-12:
        return np.full_like(vec, 0.5)
    return 0.5 + 0.5 * vec / m


def to_png(row: np.ndarray, side: int, mode: str) -> Image.Image:
    img = rescale_mnist(row) if mode == "mnist" else rescale_signed(row)
    img = (img.reshape(side, side) * 255.0).astype(np.uint8)
    return Image.fromarray(img)


def load_principles(npy_path: Path,
                    weight_row: np.ndarray | None) -> tuple[np.ndarray, int]:
    arr = np.load(npy_path)
    if arr.ndim != 2:
        raise ValueError(f"{npy_path}: expected 2D (G, dim), got {arr.shape}")
    dim = arr.shape[1]
    side = int(round(dim ** 0.5))
    if side * side != dim:
        raise ValueError(f"{npy_path}: dim {dim} is not a perfect square")
    if weight_row is not None:
        if weight_row.shape != (dim,):
            raise ValueError(
                f"{npy_path}: weight row shape {weight_row.shape} != ({dim},)")
        arr = np.nan_to_num(arr / weight_row, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, side


def render_file(npy_path: Path, out_dir: Path, mode: str, upscale: int,
                weight_row: np.ndarray | None = None) -> int:
    arr, side = load_principles(npy_path, weight_row)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(arr):
        im = to_png(row, side, mode)
        if upscale > 1:
            im = im.resize((side * upscale, side * upscale), Image.NEAREST)
        im.save(out_dir / f"principle{i:03d}.png")
    return len(arr)


def make_grid(tiles: list[Image.Image], cols: int, pad: int = 1,
              bg: int = 0) -> Image.Image:
    if not tiles:
        raise ValueError("no tiles to assemble")
    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    W = cols * tw + (cols + 1) * pad
    H = rows * th + (rows + 1) * pad
    grid = Image.new("L", (W, H), bg)
    for idx, im in enumerate(tiles):
        r, c = divmod(idx, cols)
        grid.paste(im, (pad + c * (tw + pad), pad + r * (th + pad)))
    return grid


def tiles_from_file(npy_path: Path, mode: str, upscale: int,
                    weight_row: np.ndarray | None) -> list[Image.Image]:
    arr, side = load_principles(npy_path, weight_row)
    out = []
    for row in arr:
        im = to_png(row, side, mode)
        if upscale > 1:
            im = im.resize((side * upscale, side * upscale), Image.NEAREST)
        out.append(im)
    return out


def load_weight_matrix(model_path: Path, layer_idx: int) -> np.ndarray:
    from safetensors import safe_open
    # Try both naming schemes: fc{N+1} (1-indexed) and linears.{N} (0-indexed).
    candidates = [f"fc{layer_idx + 1}.weight", f"linears.{layer_idx}.weight",
                  f"fc{layer_idx}.weight"]
    with safe_open(str(model_path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for key in candidates:
            if key in keys:
                return f.get_tensor(key).numpy()
        raise SystemExit(
            f"{model_path}: none of {candidates} found. "
            f"Available: {sorted(keys)}")


NEURON_RE = re.compile(r"neuron(\d+)_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help=".npy file or folder of *_principles_*.npy files")
    ap.add_argument("-o", "--out", required=True, help="Output directory")
    ap.add_argument("--mode", choices=["mnist", "signed"], default="mnist")
    ap.add_argument("--method", default="mean",
                    help="When --path is a folder, match *_principles_<method>.npy")
    ap.add_argument("--upscale", type=int, default=1,
                    help="Nearest-neighbor upscale factor (default 1 = 28x28)")
    ap.add_argument("--weights", default=None,
                    help="Optional .safetensors model. When given, each row is "
                         "divided element-wise by fc{layer}.weight[neuron] "
                         "before rescaling (matches the HTML viewer's "
                         "'Backwards' panel).")
    ap.add_argument("--layer", type=int, default=0,
                    help="Layer index to pull the weight row from when "
                         "--weights is set (default 0)")
    ap.add_argument("--montage", action="store_true",
                    help="Render a single composite PNG instead of one PNG "
                         "per principle. In folder mode, each row is one "
                         "neuron and each column is a principle index.")
    ap.add_argument("--montage-cols", type=int, default=None,
                    help="Override columns in --montage single-file mode "
                         "(default: ceil(sqrt(G))).")
    ap.add_argument("--neuron", type=int, default=None,
                    help="Neuron index for single-file mode when --weights "
                         "is set. If omitted, parsed from the filename "
                         "(neuron<K>_).")
    args = ap.parse_args()

    src = Path(args.path)
    out_root = Path(args.out)

    weight_mat = None
    if args.weights:
        weight_mat = load_weight_matrix(Path(args.weights), args.layer)
        print(f"Loaded fc{args.layer}.weight {weight_mat.shape} from {args.weights}")

    if src.is_file():
        wrow = None
        if weight_mat is not None:
            n_idx = args.neuron
            if n_idx is None:
                m = NEURON_RE.search(src.name)
                if not m:
                    raise SystemExit(
                        f"Cannot infer neuron index from {src.name}; "
                        f"pass --neuron")
                n_idx = int(m.group(1))
            if n_idx >= weight_mat.shape[0]:
                raise SystemExit(
                    f"neuron {n_idx} >= fc{args.layer} out_dim "
                    f"{weight_mat.shape[0]}")
            wrow = weight_mat[n_idx]
        if args.montage:
            tiles = tiles_from_file(src, args.mode, args.upscale, wrow)
            cols = args.montage_cols or max(1, int(np.ceil(np.sqrt(len(tiles)))))
            grid = make_grid(tiles, cols)
            out_root.parent.mkdir(parents=True, exist_ok=True)
            out_path = out_root if out_root.suffix.lower() == ".png" \
                else out_root / "montage.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            grid.save(out_path)
            print(f"Wrote montage {grid.size} ({len(tiles)} tiles) to {out_path}")
            return
        n = render_file(src, out_root, args.mode, args.upscale, wrow)
        print(f"Wrote {n} PNGs to {out_root}")
        return

    if not src.is_dir():
        raise SystemExit(f"Not a file or directory: {src}")

    files = sorted(src.glob(f"*_principles_{args.method}.npy"))
    if not files:
        raise SystemExit(f"No *_principles_{args.method}.npy files in {src}")

    if args.montage:
        rows_tiles: list[list[Image.Image]] = []
        labels: list[str] = []
        max_cols = 0
        for f in files:
            m = NEURON_RE.search(f.name)
            sub = f"neuron{int(m.group(1)):03d}" if m else f.stem
            wrow = None
            if weight_mat is not None:
                if not m:
                    print(f"  [skip] {f.name}: no neuron index in filename")
                    continue
                n_idx = int(m.group(1))
                if n_idx >= weight_mat.shape[0]:
                    print(f"  [skip] {f.name}: neuron {n_idx} out of range")
                    continue
                wrow = weight_mat[n_idx]
            tiles = tiles_from_file(f, args.mode, args.upscale, wrow)
            rows_tiles.append(tiles)
            labels.append(sub)
            max_cols = max(max_cols, len(tiles))
            print(f"  {labels[-1]}: {len(tiles)} principles")
        if not rows_tiles:
            raise SystemExit("no rows to render")
        tw, th = rows_tiles[0][0].size
        blank = Image.new("L", (tw, th), 0)
        flat: list[Image.Image] = []
        for tiles in rows_tiles:
            flat.extend(tiles)
            flat.extend([blank] * (max_cols - len(tiles)))
        grid = make_grid(flat, max_cols)
        out_path = out_root if out_root.suffix.lower() == ".png" \
            else out_root / "montage.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(out_path)
        print(f"\nWrote montage {grid.size} "
              f"({len(rows_tiles)} neurons x {max_cols} principles) "
              f"to {out_path}")
        return

    total_png = 0
    for f in files:
        m = NEURON_RE.search(f.name)
        sub = f"neuron{int(m.group(1)):03d}" if m else f.stem
        wrow = None
        if weight_mat is not None:
            if not m:
                print(f"  [skip] {f.name}: no neuron index in filename")
                continue
            n_idx = int(m.group(1))
            if n_idx >= weight_mat.shape[0]:
                print(f"  [skip] {f.name}: neuron {n_idx} out of range")
                continue
            wrow = weight_mat[n_idx]
        n = render_file(f, out_root / sub, args.mode, args.upscale, wrow)
        total_png += n
        print(f"  {sub}: {n} principles")
    print(f"\nWrote {total_png} PNGs across {len(files)} neurons to {out_root}")


if __name__ == "__main__":
    main()
