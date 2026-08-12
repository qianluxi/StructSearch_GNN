# AGENTS.md

# StructSearch_v2 Development Guide

## Project Overview

StructSearch_v2 is an AI-powered structural drawing retrieval system
(remote repo: `qianluxi/StructSearch_GNN`). It converts engineering CAD
drawings (DXF) into graphs, learns graph embeddings with a GNN, and supports
structural-system classification (shear wall / frame-shear / frame) plus
similarity retrieval over a vector index.

The canonical pipeline:

```
DXF
  -> dxf_parser.process_dxf_file()          # List[NodeDict]
  -> FeatureEngineering.build()             # node features (N, 9) + aligned pos (N, 2)
  -> GraphBuilder.build()                   # torch_geometric Data (x, edge_index, ...)
  -> GraphStatistics.build_feature_vector() # graph-level stats (11,)
  -> StructureDataset                      # InMemoryDataset, standardized features
  -> GraphEncoder                          # (embedding, logits)
  -> Embedder / vector index               # similarity search
```

This is a research-oriented project. Readability and maintainability matter
more than clever code.

---

# Technology Stack

- Python 3.9+ (bytecode in repo built with 3.9; keep compatibility)
- PyTorch + PyTorch Geometric (PyG)
- NumPy, SciPy, scikit-learn
- ezdxf + shapely (DXF parsing and geometry; `dxfgrabber` in
  `requirements.txt` is legacy and not used by the current code)
- matplotlib / seaborn (visualization)
- FAISS or Milvus (optional vector backends)

Avoid introducing unnecessary dependencies. When you touch dependencies,
update `requirements.txt` and `setup.py` together.

---

# Repository Layout

```
StructSearch_v2/
  configs/            # YAML configs (retrieval.yaml is the primary one)
  src/
    datasets/         # DXF parsing, features, graph construction, dataset
    models/           # GraphEncoder, backbones/, heads/
    training/         # train.py, split_dataset.py, evaluate.py (+ placeholders)
    retrieval/        # retrieval entry points (currently empty, implement here)
    retrieval/R/      # legacy working retrieval implementation (reference)
    analysis/         # embedding metrics, retrieval benchmark, visualizations
    tools/visualize/  # t-SNE / PCA / confusion / distance matrix scripts
    utils/            # config_loader.py, dxf_utils.py
  data/
    raw/ raw_dxf/     # original DXF drawings (never modify)
    processed/        # cached PyG graph data
    splits/ feature_index/
  tests/              # unit tests
  visualize/          # duplicate of legacy visualization scripts
  checkpoints/        # trained model artifacts
  fig/                # generated figures (output)
  experiments/        # embeddings, logs, checkpoints (output)
```

Follow the existing architecture. Do not create random folders. When a legacy
and a current module overlap (e.g. `src/retrieval/R/` vs `src/retrieval/`),
put new code in the non-legacy location and treat `R/` as a reference only.

---

# Core Module Interfaces

## 1. Data pipeline (`src/datasets/`)

### dxf_parser.py

- `process_dxf_file(path) -> List[NodeDict]`
- `load_polygons_from_dxf(path, close_threshold=1.0) -> List[Polygon]`
- `classify_node(r, thickness) -> "Column" | "ShortWall" | "Wall"`

`NodeDict` keys (see `schemas.py`):

```python
{
    "centroid": (x, y),
    "area": float,
    "Lx": float,           # span along x after alignment
    "Ly": float,           # span along y after alignment
    "long_leg_length": float,
    "long_leg_thickness": float,
    "r": float,            # slenderness ratio (long/short)
    "type": int,           # 0=Column, 1=ShortWall, 2=Wall
}
```

Type rules currently implemented in `classify_node`:

- thickness >= 310 and r > 4, or r >= 8 -> Wall
- r >= 4 -> ShortWall
- otherwise -> Column

### feature_engineering.py

- `FeatureEngineering().build(nodes) -> (node_features, aligned_pos)`
  - `node_features`: float32 `(N, 9)`
  - `aligned_pos`: `(N, 2)`, centroid-centered + PCA-rotated (rotation invariant)

9-D node feature layout (do not change without updating all consumers):

```text
0-2   type one-hot (Column, ShortWall, Wall)
3     area
4     L_long   (max(Lx, Ly))
5     L_short  (min(Lx, Ly))
6     long_leg_length
7     long_leg_thickness
8     r
```

### graph_builder.py

- `GraphBuilder(connect_mode="knn", k=6, radius=None).build(node_features, pos)
  -> torch_geometric.data.Data` with `x`, `edge_index`, `edge_attr` (distance), `pos`
- Supports `"knn"` and `"radius"` connect modes; never let k exceed N-1.

### graph_statistics.py

- `GraphStatistics(nodes, aligned_pos=None).build_feature_vector() -> np.ndarray (11,)`

11-D graph feature layout:

```text
0-2   type_ratio P(G)        (Column / ShortWall / Wall)
3-5   type_area_ratio Q(G)
6     log1p(total_area)
7     area CV
8     area entropy
9-10  nearest-neighbor distance mean / std
```

### structure_dataset.py

- `StructureDataset(root, k=6, connect_mode="knn", radius=None)` inherits
  `InMemoryDataset`; `root` must contain `raw/*.dxf`.
- Each `Data` object carries:
  - `x_raw`, `x` (raw / standardized node features)
  - `edge_index`, `edge_attr`, `pos`
  - `y` (label), `filename`
  - `graph_feature_raw`, `graph_feature` (raw / standardized 11-D stats)
- Standardization stats are saved with the processed file; use
  `inverse_transform(normalized_tensor)` / `inverse_node_transform(...)` to
  recover raw values. Keep the saved dict keys stable
  (`data`, `slices`, `graph_mean`, `graph_std`, `node_mean`, `node_std`).
- Labels are derived from the filename via the class map:

```python
{'剪力墙': 0, '框剪': 1, '框架': 2,
 'WALL': 0, 'FRAME_SHEAR': 1, 'FRAME': 2}
```

Files whose names do not match any key are skipped.

## 2. Models (`src/models/`)

### model.py (primary encoder)

- `GraphEncoder(node_dim=9, hidden_dim=64, emb_dim=128, num_classes=3)`
- `forward(x, edge_index, batch) -> (embedding, logits)`
  - 3 x `SAGEConv` + ReLU -> `global_mean_pool` -> `Linear(hidden_dim, emb_dim)`
  - embedding `(B, emb_dim)`, logits `(B, num_classes)`
- Keep the `(embedding, logits)` tuple contract; downstream code and tests
  rely on it.

### backbones/ and heads/

- `GINBackbone(cfg)` reads `cfg.model.hidden_dim`, `cfg.model.num_layers`,
  `cfg.model.num_node_features`; GCN / GraphTransformer are placeholders.
- `ClassifierHead(in_dim, num_classes)`, `ProjectionHead(in_dim, out_dim)`
  (projection head normalizes output for cosine similarity).
- Prefer `GraphEncoder` for new work unless a backbone is explicitly required.

## 3. Training (`src/training/`)

- `train.py` is the runnable training entry point:
  `StructureDataset(root="data")` -> `split_dataset(n, train_ratio=0.8)` ->
  PyG `DataLoader` -> `GraphEncoder` + `CrossEntropyLoss` (50 epochs) ->
  saves `checkpoints/model.pt` as `{"state_dict", "emb_dim", "num_classes"}`.
- `split_dataset(n_samples, train_ratio=0.8)` returns shuffled `(train_idx, val_idx)`.
- `pretrain.py` / `finetune.py` are placeholders; do not treat them as
  functional unless you implement them.

## 4. Retrieval (`src/retrieval/`)

The top-level `src/retrieval/*.py` files are currently empty stubs. The working
legacy implementation lives in `src/retrieval/R/` and defines the intended API:

- `Embedder(model, device=None)`:
  - `extract_one(data) -> (emb_dim,)` numpy
  - `extract_batch(dataset, batch_size=32) -> (N, emb_dim)` numpy
  - `extract(data)` auto-dispatches; `get_embedding` / `embed` are aliases
- `index_builder.build_index(cfg_path)` -> saves
  `embeddings.npy`, `filenames.npy`, `labels.npy` (GNN embedding concatenated
  with stats features)
- `SearchEngine.search(dxf_path)` -> classification result + Top-K similar cases
- `processor.process_single_dxf(filepath)` -> model-ready PyG `Data`

When implementing the stubs, mirror these interfaces so callers do not break.

## 5. Analysis & Visualization (`src/analysis/`, `src/tools/visualize/`)

- `embedding_metrics.evaluate_embedding(embeddings, labels)` returns
  silhouette score, Davies-Bouldin index, Calinski-Harabasz score, and
  Top-1/3/5 retrieval accuracy.
- `retrieval_benchmark/run_benchmark.run()` produces t-SNE, similarity heatmap,
  centroid distance, variance, retrieval accuracy curve, and case figures under
  `fig/benchmark/`.
- Visual scripts live in `src/tools/visualize/` (and a legacy copy in
  `visualize/`); prefer the `src/tools/visualize/` versions for new work.

## 6. Configuration (`configs/`, `src/utils/`)

- `utils.config_loader.load_config(yaml_path) -> SimpleNamespace` (nested
  `cfg.model.hidden_dim` style access).
- `configs/retrieval.yaml` is the primary config: device, data roots,
  checkpoint paths, model params, embedding dim.
- Put tunable parameters in YAML. Avoid hard-coded paths and hyperparameters.
- `src/config.py` holds legacy module-level globals; prefer YAML config in new
  code.

---

# Interface Rules (do not break these)

1. Keep the node feature dimension at 9 and the graph feature dimension at 11
   unless every consumer is updated at the same time:
   `FeatureEngineering`, `GraphEncoder(node_dim=...)`, `StructureDataset`
   normalization, and index concatenation.
2. Keep labels stable: 0 = shear wall (剪力墙), 1 = frame-shear (框剪),
   2 = frame (框架).
3. Keep `GraphEncoder.forward` returning `(embedding, logits)`.
4. Dataset objects expose both raw (`x_raw`, `graph_feature_raw`) and
   standardized (`x`, `graph_feature`) tensors; never feed raw tensors to the
   model by default.
5. Extend existing classes/interfaces instead of creating parallel modules.
   Consolidate duplicates (e.g. `src/retrieval/` vs `src/retrieval/R/`,
   `src/tools/visualize/` vs `visualize/`) rather than adding more.
6. When a model, dataset, or config default changes, update the corresponding
   tests and configs in the same commit.

---

# Coding Principles

Prefer:

- clear, modular code
- reusable functions with explicit typing
- meaningful variable names
- vectorized NumPy / batch operations for large DXF sets
- caching and incremental processing

Avoid:

- duplicated code
- giant functions
- magic numbers
- hard-coded paths
- global mutable state

---

# Data Safety

Never modify:

- raw DXF files (`data/raw/`, `data/raw_dxf/`)
- benchmark datasets
- pretrained model checkpoints

Generated files go into:

- `experiments/`, `logs/`, `fig/`, `checkpoints/`

Never overwrite original data unless explicitly requested.

---

# Machine Learning Guidelines

Models must be reproducible, deterministic when possible, and configurable.

- Fix random seeds (the dataset pipeline already uses seed 42 and deterministic
  cuDNN flags; keep that behavior).
- Keep model / trainer / dataset / evaluation separated.
- Prefer configurable dimensions and thresholds over literals.

---

# Documentation & Testing

- Every public class/function should have a docstring.
- Complex algorithms need concise comments: purpose, inputs, outputs.
- Before a task is complete, verify:
  - code runs without syntax errors
  - existing functionality stays compatible
  - imports are valid
  - configuration files remain consistent
- Existing tests live in `tests/`; some reference the older `GraphEncoder`
  signature (`in_dim=`) and need updating when that interface moves — keep
  tests aligned with the current `node_dim` signature.

---

# Git Workflow

- Prefer small, atomic commits.
- Suggested prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `perf:`.
- Branch convention follows the project history: `vX.Y` version branches
  (currently up to `v1.8`).
- Never rewrite project history; do not force-push shared branches.
- Commit code that runs; avoid committing large binaries (DXF datasets,
  checkpoints, figures) unless explicitly requested — use `.gitignore` /
  Git LFS for those.

---

# Current State & Known Gaps

When working in this repo, be aware of the following:

- `main.py` imports `src.training.trainer.Trainer`, which no longer exists;
  the runnable entry point is `src/training/train.py`.
- `src/retrieval/*.py` are empty stubs; the legacy implementation in
  `src/retrieval/R/` is the reference API.
- `pretrain.py` / `finetune.py` are placeholders.
- `requirements.txt` lists `dxfgrabber`, but the code uses `ezdxf` + `shapely`;
  update dependencies when you touch them.
- Some README/docs files have GBK/UTF-8 encoding issues; write new docs as
  UTF-8.

---

# Preferred Response Style

When implementing a feature:

1. Explain the implementation plan.
2. Modify the necessary files.
3. Summarize the changes.
4. Mention any assumptions.
5. Mention possible follow-up improvements.

Avoid unnecessary architectural changes; favor incremental improvements over
large rewrites. When uncertain, ask before making destructive changes.
