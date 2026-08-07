# AGENTS.md

# StructSearch_v3 Development Guide

## Project Overview

StructSearch_v3 is an AI-powered structural drawing retrieval system.

The project converts engineering CAD drawings (DXF) into graph representations,
extracts graph embeddings using Graph Neural Networks (GNN), and performs
similarity retrieval through vector search.

Current objectives:

- DXF → Graph conversion
- Structural feature extraction
- Graph embedding learning
- Structural classification
- Similarity retrieval
- Retrieval benchmarking
- Engineering visualization

This is a research-oriented project. Code readability and maintainability are
more important than writing overly clever code.

---

# Technology Stack

Primary language:

- Python 3.11+

Libraries:

- PyTorch
- PyTorch Geometric
- NumPy
- SciPy
- NetworkX
- ezdxf
- shapely
- scikit-learn
- Milvus (optional)
- FAISS (optional)

Avoid introducing unnecessary dependencies.

---

# Project Structure

Typical directory layout:

```
StructSearch_v3/

configs/
src/
    datasets/
    models/
    retrieval/
    training/
    analysis/
    visualization/
tools/
scripts/
data/
models/
docs/
```

Follow the existing architecture whenever possible.

Do not create random folders.

---

# Coding Principles

Prefer:

- clear code
- modular design
- reusable functions
- explicit typing
- meaningful variable names

Avoid:

- duplicated code
- giant functions
- magic numbers
- hard-coded paths
- global mutable state

---

# File Modification Rules

Prefer modifying existing modules instead of creating duplicate files.

When adding functionality:

- reuse existing classes
- extend existing interfaces
- avoid rewriting working code

Do not rename files unless necessary.

---

# Data Safety

Never modify:

- raw DXF files
- benchmark datasets
- pretrained model checkpoints

Generated files should be written into:

- outputs/
- experiments/
- logs/

Never overwrite original data unless explicitly requested.

---

# Configuration

Parameters should be placed in configuration files whenever possible.

Avoid hard-coded values.

Examples:

- embedding dimension
- learning rate
- batch size
- retrieval top-k
- thresholds

---

# Machine Learning Guidelines

Models should be:

- reproducible
- deterministic when possible
- configurable

Random seeds should be fixed unless experimentation requires randomness.

Prefer modular model definitions.

Separate:

- model
- trainer
- dataset
- evaluation

---

# Retrieval Pipeline

Typical workflow:

DXF
    ↓
Graph Construction
    ↓
Node Feature Engineering
    ↓
Graph Encoder
    ↓
Embedding
    ↓
Vector Database
    ↓
Similarity Search

Do not break this pipeline without clear justification.

---

# Performance

When processing large DXF datasets:

Prefer:

- vectorized NumPy operations
- batch processing
- caching
- incremental processing

Avoid unnecessary repeated parsing.

---

# Documentation

Every public class should have a docstring.

Complex algorithms should include concise comments explaining:

- purpose
- inputs
- outputs

Avoid redundant comments.

---

# Testing

Before considering a task complete:

- code runs without syntax errors
- existing functionality remains compatible
- imports are valid
- configuration files remain consistent

Do not remove existing features unless requested.

---

# Git Workflow

Prefer small, atomic commits.

Suggested commit style:

- feat:
- fix:
- refactor:
- docs:
- test:
- perf:

Never rewrite project history.

---

# Preferred Response Style

When implementing a feature:

1. Explain the implementation plan.
2. Modify the necessary files.
3. Summarize the changes.
4. Mention any assumptions.
5. Mention possible follow-up improvements.

Avoid making unnecessary architectural changes.

---

# General Principles

Favor incremental improvements over large rewrites.

Respect the existing project structure.

Maintain backward compatibility whenever practical.

When uncertain, ask before making destructive changes.