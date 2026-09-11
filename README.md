<div align="center">
  <img src="https://raw.githubusercontent.com/OlivierBeq/chemsplit/refs/heads/main/graphics/chemsplit_logo.svg" alt="chemsplit logo" width="300">

  # ✂️ chemsplit

  [![PyPI version](https://img.shields.io/pypi/v/chemsplit.svg)](https://pypi.org/project/chemsplit/)
  [![Supported Python versions](https://img.shields.io/pypi/pyversions/chemsplit.svg)](https://pypi.org/project/chemsplit/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Tests](https://github.com/OlivierBeq/chemsplit/actions/workflows/ci.yml/badge.svg)](https://github.com/OlivierBeq/chemsplit/actions/workflows/ci.yml)
  [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

A self-contained, scikit-learn-compatible Python library of dataset-splitting strategies for cheminformatics machine learning. `chemsplit` implements **53 splitting strategies across nine families** — baseline, scaffold, similarity, embedding, property, lineage, task, biomolecular, and protocol splitters — behind one coherent, deterministic API, plus a leakage-audit module and a set of reference/synthetic datasets to try them on.

## ✨ Features

- 🧩 **9 families, 53 strategies** — from a plain random split to scaffold-tree pruning, Butina/spectral clustering, UMAP-space holdouts, temporal and provenance cuts, protein-family and binding-site holdouts, drug-target cold-start benchmarks, and full CV/nested-CV protocol wrappers.
- 🎯 **Deterministic by construction** — every splitter accepts a `random_state` and produces bit-identical output regardless of record order or `n_jobs`, checked continuously by a golden-file regression suite and Hypothesis property tests.
- 🛡️ **Contract-checked results** — every `SplitResult` is validated against five structural invariants (index coverage, disjointness, group-label consistency, JSON round-tripping of `params`, id format) before it ever reaches your code.
- 🔍 **Built-in leakage auditing** — `chemsplit.audit` reports nearest-neighbour similarity, adversarial-validation AUC, exact/scaffold/ring-system overlap, and property/label shift between train and test.
- 🧪 **Chemistry-native featurization** — ECFP/FCFP/MACCS/Avalon/atom-pair/topological-torsion fingerprints and physicochemical descriptors, behind a pluggable `Featurizer` protocol for your own.
- 💻 **CLI included** — run, audit, and list any registered splitter without writing a line of Python.
- 📚 **One example notebook per family** — runnable, narrated walkthroughs of every splitter class under [`notebooks/`](notebooks/).

## 📦 Installation

```bash
pip install chemsplit
```

Optional extras enable additional splitters and featurizers:

| Extra | Enables |
|---|---|
| `chemsplit[umap]` | `UMAPClusterSplitter` (UMAP embedding + clustering) |
| `chemsplit[hdbscan]` | `DensityClusterSplitter` (HDBSCAN density clustering) |
| `chemsplit[ga]` | `SIMPDSplitter` (genetic-algorithm pseudo-time optimization, via `deap`) |
| `chemsplit[bio]` | Protein-sequence splitters with accelerated alignment (`biopython`, `parasail`) |
| `chemsplit[mmpa]` | `MatchedMolecularSeriesSplitter` matched-series extraction |
| `chemsplit[all]` | Everything above |

> **Note:** every extra has a dependency-free fallback where one makes sense (e.g. a Hamming-distance fallback for sequence identity without `bio`) — an extra buys you a better implementation, not a hard requirement.

## 🛠️ Requirements

- Python 3.11 – 3.13
- [RDKit](https://www.rdkit.org/) (installed automatically as a core dependency — no separate conda step needed)

## 💡 Usage

### Quickstart

```python
from chemsplit import datasets, get_splitter

fx = datasets.make_scaffold_families(n_scaffolds=10, per_scaffold=15)

splitter = get_splitter("scaffold_tree", train_size=0.8, test_size=0.2)
result = splitter.split_result(fx.smiles)[0]

train_smiles = [fx.smiles[i] for i in result.train]
test_smiles = [fx.smiles[i] for i in result.test]
print(f"{len(train_smiles)} train / {len(test_smiles)} test records")
```

Every splitter is resolved the same way, by its `splitter_id` (see `chemsplit.list_splitters()` for the full table) or by importing the class directly:

```python
from chemsplit import ButinaSplitter

splitter = ButinaSplitter(cutoff=0.4, train_size=0.7, test_size=0.3, random_state=0)
```

<details>
<summary><strong>🔍 Auditing a split for leakage</strong></summary>

```python
from chemsplit import audit_split

report = audit_split(result, fx.smiles)
print(report.summary())
```
```text
chemsplit LeakageReport
------------------------
n_train=120  n_valid=0  n_test=30  n_discard=0
max cross-partition similarity: 0.4375
median NN similarity (test->train): 0.6667
exact duplicates across partitions: 0
shared scaffolds: 0  shared ring systems: 0
adversarial AUC: 1.0000 (95% CI 1.0000-1.0000)
flags: MEDIAN_NN_ABOVE_0.6, HIGH_ADVERSARIAL_AUC, ...
```

`LeakageReport` is purely descriptive — it never fails your pipeline, it tells you what to look at.
</details>

<details>
<summary><strong>💻 The CLI</strong></summary>

```bash
# List every registered splitter, optionally filtered by family
chemsplit list --family scaffold

# Run a splitter over a CSV and write the split to disk
chemsplit split --splitter scaffold_tree --input data.csv --smiles-col smiles \
  --train-size 0.8 --test-size 0.2 --seed 0 --out split.json

# Audit that split for leakage
chemsplit audit --split split.json --input data.csv --smiles-col smiles --out report.json

chemsplit --help
```
</details>

## 🧭 Design principles

1. **Determinism.** Same inputs + same `random_state` ⇒ byte-identical outputs, on any platform, any CPU count, any `n_jobs`.
2. **Explicitness.** No silent fallbacks. If a requested configuration is infeasible, raise, never approximate.
3. **Honesty.** Every splitter's docstring discloses its pitfalls with the same prominence as its advantages.
4. **Composability.** Every group-forming splitter exposes its group labels, so any grouping can be fed to any protocol wrapper.
5. **scikit-learn compatibility.** `.split()` is drop-in usable in `cross_val_score`, `GridSearchCV(cv=...)`, and `cross_validate`.

## 📚 Learn more

One narrated, runnable notebook per family, under [`notebooks/`](notebooks/):

- [`baseline.ipynb`](notebooks/baseline.ipynb) — uninformed and gold-standard controls
- [`scaffold.ipynb`](notebooks/scaffold.ipynb) — chemotype-aware generalization tests
- [`similarity.ipynb`](notebooks/similarity.ipynb) — fingerprint-clustering holdouts
- [`embedding.ipynb`](notebooks/embedding.ipynb) — latent-space holdouts
- [`property.ipynb`](notebooks/property.ipynb) — label-shift and distributional stress tests
- [`lineage.ipynb`](notebooks/lineage.ipynb) — date and provenance-aware holdouts
- [`biomolecular.ipynb`](notebooks/biomolecular.ipynb) — protein-axis holdouts
- [`task.ipynb`](notebooks/task.ipynb) — drug-target interaction benchmarks
- [`protocol.ipynb`](notebooks/protocol.ipynb) — cross-validation and evaluation harnesses

## 📄 License

This project is licensed under the [MIT License](LICENSE).
</content>
