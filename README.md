<div align="center">
  <img src="https://raw.githubusercontent.com/OlivierBeq/chemsplit/refs/heads/main/graphics/chemsplit_logo.svg" alt="chemsplit logo" width="300">

  # ✂️ chemsplit

  [![PyPI version](https://img.shields.io/pypi/v/chemsplit.svg)](https://pypi.org/project/chemsplit/)
  [![Supported Python versions](https://img.shields.io/pypi/pyversions/chemsplit.svg)](https://pypi.org/project/chemsplit/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Tests](https://github.com/OlivierBeq/chemsplit/actions/workflows/ci.yml/badge.svg)](https://github.com/OlivierBeq/chemsplit/actions/workflows/ci.yml)
  [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

A self-contained, scikit-learn-compatible Python library of dataset-splitting strategies for cheminformatics machine learning. `chemsplit` implements **56 splitting strategies across nine families** -- baseline, scaffold, similarity, embedding, property, lineage, task, biomolecular, and protocol splitters -- behind one coherent, deterministic API, plus a leakage-audit module and a set of reference/synthetic datasets to try them on.

## ✨ Features

- 🧩 **9 families, 56 strategies** -- from a plain random split to scaffold-tree pruning, Butina/spectral clustering, UMAP-space holdouts, temporal and provenance cuts, protein-family and binding-site holdouts, drug-target cold-start benchmarks, and full CV/nested-CV protocol wrappers.
- 🎯 **Deterministic by construction** -- every splitter accepts a `random_state` and produces bit-identical output regardless of record order or `n_jobs`, checked continuously by a golden-file regression suite and Hypothesis property tests.
- 🛡️ **Contract-checked results** -- every `SplitResult` is validated against five structural invariants (index coverage, disjointness, group-label consistency, JSON round-tripping of `params`, id format) before it ever reaches your code.
- 🔍 **Built-in leakage auditing** -- `chemsplit.audit` reports nearest-neighbour similarity, adversarial-validation AUC, exact/scaffold/ring-system overlap, and property/label shift between train and test.
- 🧪 **Chemistry-native featurization** -- ECFP/FCFP/MACCS/Avalon/atom-pair/topological-torsion fingerprints and physicochemical descriptors, behind a pluggable `Featurizer` protocol for your own.
- 💻 **CLI included** -- run, audit, and list any registered splitter without writing a line of Python.
- 📚 **One example notebook per family** -- runnable, narrated walkthroughs of every splitter class under [`notebooks/`](notebooks/).

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
| `chemsplit[som]` | `SelfOrganizingMapSplitter` (self-organizing maps via `ksom` and PyTorch) |
| `chemsplit[all]` | Everything above |

> **Note:** every extra has a dependency-free fallback where one makes sense (e.g. a Hamming-distance fallback for sequence identity without `bio`) -- an extra buys you a better implementation, not a hard requirement.

## 🛠️ Requirements

- Python 3.11 - 3.13
- [RDKit](https://www.rdkit.org/) (installed automatically as a core dependency -- no separate conda step needed)

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

`LeakageReport` is purely descriptive -- it never fails your pipeline, it tells you what to look at.
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

1. **Determinism.** Same inputs + same `random_state` ⇒ byte-identical outputs, on any platform, any CPU count, any `n_jobs`. An integer `random_state` is reconstructible by anyone who knows it -- use `random_state=None` for holdouts that must stay secret.
2. **Explicitness.** No silent fallbacks. If a requested configuration is infeasible, raise, never approximate.
3. **Honesty.** Every splitter's docstring discloses its pitfalls with the same prominence as its advantages.
4. **Composability.** Every group-forming splitter exposes its group labels, so any grouping can be fed to any protocol wrapper.
5. **scikit-learn compatibility.** `.split()` is drop-in usable in `cross_val_score`, `GridSearchCV(cv=...)`, and `cross_validate`.

## 📚 Learn more

One narrated, runnable notebook per family, under [`notebooks/`](notebooks/):

- [`baseline.ipynb`](notebooks/baseline.ipynb) -- uninformed and gold-standard controls
- [`scaffold.ipynb`](notebooks/scaffold.ipynb) -- chemotype-aware generalization tests
- [`similarity.ipynb`](notebooks/similarity.ipynb) -- fingerprint-clustering holdouts
- [`embedding.ipynb`](notebooks/embedding.ipynb) -- latent-space holdouts
- [`property.ipynb`](notebooks/property.ipynb) -- label-shift and distributional stress tests
- [`lineage.ipynb`](notebooks/lineage.ipynb) -- date and provenance-aware holdouts
- [`biomolecular.ipynb`](notebooks/biomolecular.ipynb) -- protein-axis holdouts
- [`task.ipynb`](notebooks/task.ipynb) -- drug-target interaction benchmarks
- [`protocol.ipynb`](notebooks/protocol.ipynb) -- cross-validation and evaluation harnesses

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📋 Available splitters

<details>
<summary><strong>Full list of splitters (56 classes)</strong> -- click to expand</summary>

| Family | Class | `splitter_id` | Description | Strictness | Group-forming | Needs | Extra |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|
| Baseline | `RandomSplitter` | `random` | Uniform random (or fixed-order) permutation | 🔴 optimistic | -- | -- | -- |
| Baseline | `StratifiedRandomSplitter` | `stratified_random` | Random split stratified on the label | 🔴 optimistic | -- | labels | -- |
| Baseline | `KFoldSplitter` | `k_fold` | k-fold CV (optionally stratified or leave-one-out) | 🔴 optimistic | -- | -- | -- |
| Baseline | `MonteCarloSplitter` | `monte_carlo` | Repeated independent random splits | 🔴 optimistic | -- | -- | -- |
| Baseline | `PredefinedSplitter` | `predefined` | Wraps an externally supplied assignment | 🟠 moderate | -- | -- | -- |
| Scaffold | `MurckoScaffoldSplitter` | `murcko_scaffold` | Groups by Murcko scaffold | 🟠 moderate | :heavy_check_mark: | -- | -- |
| Scaffold | `GenericScaffoldSplitter` | `generic_scaffold` | Groups by generic (all-carbon) scaffold framework | 🟢 strict | :heavy_check_mark: | -- | -- |
| Scaffold | `ScaffoldTreeSplitter` | `scaffold_tree` | Groups by scaffold-tree node at a chosen pruning level | 🟢 strict | :heavy_check_mark: | -- | -- |
| Scaffold | `RingSystemSplitter` | `ring_system` | Groups by shared ring systems, transitively | 🟠 moderate | :heavy_check_mark: | -- | -- |
| Scaffold | `MatchedMolecularSeriesSplitter` | `matched_molecular_series` | Groups molecules sharing a matched-pair constant context | 🟢 strict | :heavy_check_mark: | -- | `mmpa` |
| Scaffold | `ActivityCliffSplitter` | `activity_cliff` | Places activity-cliff compounds in the test set | 🚀 extrapolative | -- | labels | -- |
| Similarity | `SimilarityThresholdSplitter` | `similarity_threshold` | No test record above a similarity threshold to train | 🟢 strict | :heavy_check_mark: | -- | -- |
| Similarity | `ButinaSplitter` | `butina` | Taylor-Butina sphere-exclusion clustering | 🟢 strict | :heavy_check_mark: | -- | -- |
| Similarity | `SphereExclusionSplitter` | `sphere_exclusion` | Random-order sphere-exclusion clustering with a fixed radius | 🟢 strict | :heavy_check_mark: | -- | -- |
| Similarity | `KMeansClusterSplitter` | `k_means_cluster` | K-means clustering over fingerprint/feature space | 🟢 strict | :heavy_check_mark: | -- | -- |
| Similarity | `DensityClusterSplitter` | `density_cluster` | DBSCAN/HDBSCAN density clustering | 🟢 strict | :heavy_check_mark: | -- | -- |
| Similarity | `SpectralSplitter` | `spectral` | Spectral clustering on an affinity graph | 🚀 extrapolative | :heavy_check_mark: | -- | -- |
| Similarity | `MaxMinSplitter` | `max_min` | Greedy maximally-diverse selection (MaxMin / Kennard-Stone) | 🟠 moderate | -- | -- | -- |
| Similarity | `SPXYSplitter` | `spxy` | Kennard-Stone selection over joint feature and label distances | 🟠 moderate | -- | labels | -- |
| Similarity | `OptiSimSplitter` | `opti_sim` | OptiSim selection as cluster centres or a diverse picked set | 🟢 strict | :heavy_check_mark: | -- | -- |
| Similarity | `MaxDissimilaritySplitter` | `max_dissimilarity` | Pushes train and test to opposite regions of chemical space | 🚀 extrapolative | -- | -- | -- |
| Similarity | `PerimeterSplitter` | `perimeter` | Holds out the outskirts; trains on the dense core | 🚀 extrapolative | -- | -- | -- |
| Similarity | `LeaveOneClusterOutSplitter` | `leave_one_cluster_out` | Each cluster takes a turn as the test fold | 🚀 extrapolative | :heavy_check_mark: | -- | -- |
| Similarity | `BalancedMultiTaskSplitter` | `balanced_multi_task` | Assigns whole clusters to folds, balancing every task | 🟢 strict | :heavy_check_mark: | labels | -- |
| Embedding | `UMAPClusterSplitter` | `umap_cluster` | UMAP embedding followed by clustering | 🚀 extrapolative | :heavy_check_mark: | -- | `umap` |
| Embedding | `ProjectionSplitter` | `projection` | Linear/manifold projection, then clustering, axis cut, or grid | 🟢 strict | :heavy_check_mark: | -- | -- |
| Embedding | `LatentSpaceSplitter` | `latent_space` | Clusters in a caller-supplied embedding | 🟢 strict | :heavy_check_mark: | -- | -- |
| Property | `PropertySplitter` | `property` | Cuts along a continuous molecular property | 🟢 strict | -- | -- | -- |
| Property | `LabelExtrapolationSplitter` | `label_extrapolation` | Trains on one part of the label range, tests on another | 🚀 extrapolative | -- | labels | -- |
| Property | `StratifiedDistributionSplitter` | `stratified_distribution` | Matches the full label distribution across partitions | 🔴 optimistic | -- | labels | -- |
| Property | `MOODSplitter` | `mood` | Picks the candidate split whose train→test distances best match train→deployment | 🟢 strict | -- | -- | -- |
| Property | `AdversarialSplitter` | `adversarial` | Audits or constructs a split with a train-vs-test discriminator | 🟢 strict | -- | -- | -- |
| Lineage | `TemporalSplitter` | `temporal` | Date cut: train on the past, test on the future | 🟢 strict | -- | dates | -- |
| Lineage | `SIMPDSplitter` | `simpd` | Simulated time split optimized by a genetic algorithm | 🟢 strict | -- | labels | `ga` |
| Lineage | `SourceSplitter` | `source` | Groups by provenance (document, assay, lab, vendor, ...) | 🟢 strict | :heavy_check_mark: | -- | -- |
| Lineage | `PartySplitter` | `party` | Non-IID partition across data owners for federated evaluation | 🚀 extrapolative | :heavy_check_mark: | -- | -- |
| Task | `HiSplitter` | `hi` | Hit-identification: no test molecule similar to any train molecule | 🚀 extrapolative | :heavy_check_mark: | -- | -- |
| Task | `LoSplitter` | `lo` | Lead-optimisation: holds out analogue clusters spanning an activity range | 🟢 strict | :heavy_check_mark: | labels | -- |
| Task | `ScaffoldHopSplitter` | `scaffold_hop` | Test actives with scaffolds absent from train | 🚀 extrapolative | :heavy_check_mark: | labels | -- |
| Task | `ColdDrugSplitter` | `cold_drug` | Held-out compounds are unseen in train | 🟢 strict | :heavy_check_mark: | -- | -- |
| Task | `ColdTargetSplitter` | `cold_target` | Held-out targets are unseen in train | 🚀 extrapolative | :heavy_check_mark: | -- | -- |
| Task | `ColdPairSplitter` | `cold_pair` | Neither compound nor target of a test pair is seen in train | 🚀 extrapolative | :heavy_check_mark: | -- | -- |
| Task | `AVESplitter` | `ave` | Minimizes nearest-neighbour analogue bias (AVE) | 🟢 strict | -- | labels | `ga` |
| Task | `DecoyBenchmarkSplitter` | `decoy_benchmark` | Property-matched decoy sets or a curated benchmark partition | 🟢 strict | :heavy_check_mark: | labels | -- |
| Biomolecular | `SequenceIdentitySplitter` | `sequence_identity` | Groups protein sequences by pairwise identity | 🟢 strict | :heavy_check_mark: | -- | `bio` |
| Biomolecular | `ProteinFamilySplitter` | `protein_family` | Holds out whole target families | 🟢 strict | :heavy_check_mark: | -- | -- |
| Biomolecular | `BindingSiteSplitter` | `binding_site` | Clusters on binding-pocket composition | 🟢 strict | :heavy_check_mark: | -- | `bio` |
| Biomolecular | `DepositionDateSplitter` | `deposition_date` | Date cut for structures, pruning near-duplicate train records | 🟢 strict | -- | dates | -- |
| Biomolecular | `ComplexJointSplitter` | `complex_joint` | Jointly novel on both ligand and sequence axes | 🚀 extrapolative | -- | -- | `bio` |
| Protocol | `GroupKFoldSplitter` | `group_k_fold` | k-fold CV whose folds never split a group | 🟠 moderate | :heavy_check_mark: | -- | -- |
| Protocol | `ThreeWaySplitter` | `three_way` | Applies a splitter's criterion at both train/valid and valid/test | 🟠 moderate | -- | -- | -- |
| Protocol | `RepeatedSplitter` | `repeated` | Repeats a splitter under derived seeds | 🟠 moderate | -- | -- | -- |
| Protocol | `NestedCVSplitter` | `nested_cv` | Nested cross-validation around any outer/inner splitter | 🟠 moderate | -- | -- | -- |
| Protocol | `ExternalHoldoutSplitter` | `external_holdout` | Uses an external dataset as the entire test set | 🟠 moderate | -- | -- | -- |
| Protocol | `ApplicabilityDomainSplitter` | `applicability_domain` | Test bands at increasing distance from train | 🟠 moderate | -- | -- | -- |
| Protocol | `IntersectionSplitter` | `intersection` | Primary split in which no secondary group straddles train/test | 🟢 strict | -- | -- | -- |

</details>

Strictness ranks how hard the test set is expected to be, from 🔴 `optimistic` (random-like) through 🟠 `moderate` and 🟢 `strict` to 🚀 `extrapolative` (deliberately out-of-distribution). *Needs* lists inputs required beyond SMILES; *Extra* names the optional install extra that enables or accelerates the splitter. The same table is available at runtime via `chemsplit.list_splitters()` or `chemsplit list`.
