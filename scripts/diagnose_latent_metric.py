"""
Compare LeWM latent geometry vs dataset state geometry on PushT (HDF5).

Loads the same preprocessing as training, encodes single frames with the trained
JEPA (torch.save checkpoint), samples random pairs of dataset rows, and reports
Pearson / Spearman correlation between latent distance and state distance.

Example:
  python scripts/diagnose_latent_metric.py \\
    ckpt_path=/path/to/lewm_epoch_15_object.ckpt

Uses GPU by default (see config diagnose.device). Override with device=cpu if needed.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from jepa import JEPA
from utils import get_column_normalizer, get_img_preprocessor


def _train_namespace(cfg: DictConfig) -> DictConfig:
    """`/train/lewm` often merges under cfg.train; plain train.py uses cfg root."""
    tr = OmegaConf.select(cfg, "train")
    if tr is not None and OmegaConf.is_config(tr) and "wm" in tr:
        return tr
    if "wm" in cfg:
        return cfg
    raise RuntimeError(
        "Could not find wm (cfg.wm or cfg.train.wm). Include Hydra default /train/lewm."
    )


def _build_dataset(cfg: DictConfig):
    tc = _train_namespace(cfg)
    # pusht.yaml uses ${wm.num_preds} for num_steps; composition often breaks interpolation.
    if "data" not in cfg or "dataset" not in cfg.data:
        raise RuntimeError("cfg.data.dataset missing; check diagnose Hydra defaults.")

    with open_dict(cfg.data.dataset):
        cfg.data.dataset.num_steps = int(tc.wm.num_preds) + int(tc.wm.history_size)

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=tc.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
    return dataset


def _encode_unique_rows(
    model: JEPA,
    dataset,
    row_indices: np.ndarray,
    state_key: str,
    batch_size: int,
    device: torch.device,
    use_bf16: bool,
) -> tuple[torch.Tensor, np.ndarray]:
    """Latents z (N, D) and states s (N, state_dim) from last timestep; matches training transforms."""
    zs = []
    states = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(row_indices), batch_size):
            chunk = row_indices[start : start + batch_size]
            batch_pixels = []
            for idx in chunk:
                sample = dataset[int(idx)]
                px = sample["pixels"]
                if px.dim() == 4:
                    px = px[-1]
                elif px.dim() == 3:
                    pass
                else:
                    raise ValueError(f"Unexpected pixels shape {tuple(px.shape)}")
                batch_pixels.append(px)

                st = sample[state_key]
                if torch.is_tensor(st):
                    st_t = st[-1] if st.dim() > 1 else st
                    states.append(st_t.cpu().numpy())
                else:
                    st_a = np.asarray(st)
                    states.append(st_a[-1] if st_a.ndim > 1 else st_a)

            pixels = torch.stack(batch_pixels, dim=0).unsqueeze(1).float().to(device)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and use_bf16
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                out = model.encode({"pixels": pixels})
            emb = out["emb"][:, -1, :].float()
            zs.append(emb.cpu())

    z_cat = torch.cat(zs, dim=0)
    s_cat = np.stack(states, axis=0).astype(np.float64)
    return z_cat, s_cat


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def _resolve_device(cfg: DictConfig) -> torch.device:
    mode = str(cfg.get("device", "cuda")).lower()
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda but CUDA is not available. Use device=cpu or install a CUDA build of PyTorch."
            )
        idx = int(cfg.get("cuda_device", 0))
        return torch.device(f"cuda:{idx}")
    if mode == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device={mode!r}; use cuda, cpu, or auto.")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        r, _ = spearmanr(x, y)
        return float(r)
    except ImportError:
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        return _pearson(rx, ry)


@hydra.main(version_base=None, config_path="../config", config_name="diagnose/latent_metric")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    ckpt = Path(cfg.ckpt_path).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = _resolve_device(cfg)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print(f"Using device: {device}")

    use_bf16 = bool(cfg.get("use_bf16", True)) and device.type == "cuda"

    rng = np.random.default_rng(cfg.seed)

    dataset = _build_dataset(cfg)
    state_key = cfg.state_key
    if state_key not in dataset.column_names:
        raise KeyError(
            f"state_key={state_key!r} not in dataset columns {dataset.column_names}"
        )

    n = len(dataset)
    num_pairs = int(cfg.num_pairs)
    idx_a = rng.integers(0, n, size=num_pairs)
    idx_b = rng.integers(0, n, size=num_pairs)
    unique_idx, inverse = np.unique(
        np.concatenate([idx_a, idx_b]), return_inverse=True
    )
    inv_a, inv_b = inverse[:num_pairs], inverse[num_pairs:]

    print(f"Loading JEPA from {ckpt} …")
    try:
        model: JEPA = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(ckpt, map_location=device)
    model = model.to(device)

    print(
        f"Encoding {len(unique_idx)} unique rows "
        f"(batch_size={cfg.batch_size}, device={device}"
        f"{', bf16' if use_bf16 else ''}) …"
    )
    z_unique, s_unique = _encode_unique_rows(
        model,
        dataset,
        unique_idx,
        state_key=state_key,
        batch_size=int(cfg.batch_size),
        device=device,
        use_bf16=use_bf16,
    )
    z_a = z_unique[inv_a]
    z_b = z_unique[inv_b]
    s_a = s_unique[inv_a]
    s_b = s_unique[inv_b]

    dist_lat_l1 = (z_a - z_b).abs().sum(dim=-1).numpy()
    dist_lat_l2 = torch.linalg.vector_norm(z_a - z_b, ord=2, dim=-1).numpy()
    dist_state_l2 = np.linalg.norm(s_a - s_b, axis=1)

    r_pearson_l1 = _pearson(dist_lat_l1, dist_state_l2)
    r_pearson_l2 = _pearson(dist_lat_l2, dist_state_l2)
    r_spear_l1 = _spearman(dist_lat_l1, dist_state_l2)
    r_spear_l2 = _spearman(dist_lat_l2, dist_state_l2)

    print("\n=== Latent vs state distance ===")
    print(f"pairs: {num_pairs}, unique frames encoded: {len(unique_idx)}")
    print(f"Pearson(lat_L1, state_L2):  {r_pearson_l1:.4f}")
    print(f"Pearson(lat_L2, state_L2):  {r_pearson_l2:.4f}")
    if not np.isnan(r_spear_l1):
        print(f"Spearman(lat_L1, state_L2): {r_spear_l1:.4f}")
        print(f"Spearman(lat_L2, state_L2): {r_spear_l2:.4f}")

    out_dir = Path.cwd()
    plot_path = out_dir / "latent_vs_state.png"
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(dist_state_l2, dist_lat_l1, s=4, alpha=0.35, c="C0")
        ax.set_xlabel(r"$\|s_i - s_j\|_2$ (normalized state)")
        ax.set_ylabel(r"$\|z_i - z_j\|_1$")
        ax.set_title(f"Pearson r = {r_pearson_l1:.3f}")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"\nSaved scatter plot to {plot_path}")
        plt.close(fig)
    except ImportError:
        print("(matplotlib not installed; skip plot)")

    summary = out_dir / "latent_metric_summary.txt"
    with open(summary, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write(f"Pearson(lat_L1, state_L2):  {r_pearson_l1}\n")
        f.write(f"Pearson(lat_L2, state_L2):  {r_pearson_l2}\n")
        f.write(f"Spearman(lat_L1, state_L2): {r_spear_l1}\n")
        f.write(f"Spearman(lat_L2, state_L2): {r_spear_l2}\n")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
