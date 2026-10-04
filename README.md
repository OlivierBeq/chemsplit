<div align="center">
  <img src="https://raw.githubusercontent.com/OlivierBeq/chemsplit/refs/heads/main/graphics/chemsplit_logo.svg" alt="chemsplit logo" width="300">

  # ✂️ chemsplit

  [![PyPI version](https://img.shields.io/pypi/v/chemsplit.svg)](https://pypi.org/project/chemsplit/)
  [![Supported Python versions](https://img.shields.io/pypi/pyversions/chemsplit.svg)](https://pypi.org/project/chemsplit/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Tests](https://github.com/OlivierBeq/chemsplit/actions/workflows/ci.yml/badge.svg)](https://github.com/OlivierBeq/chemsplit/actions/workflows/ci.yml)
  [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

A self-contained, scikit-learn-compatible Python library of dataset-splitting strategies for cheminformatics machine learning. `chemsplit` implements **64 splitting strategies across nine families** -- baseline, scaffold, similarity, embedding, property, lineage, task, biomolecular, and protocol splitters -- behind one coherent, deterministic API, plus a leakage-audit module and a set of reference/synthetic datasets to try them on.

## ✨ Features

- 🧩 **9 families, 64 strategies** -- from a plain random split to scaffold-tree pruning, Butina/spectral clustering, UMAP-space holdouts, temporal and provenance cuts, protein-family and binding-site holdouts, drug-target cold-start benchmarks, and full CV/nested-CV protocol wrappers.
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

## 🎯 Choosing a split

A split is a stand-in for one deployment question: *"how will the model do on the molecules I will actually predict?"* A split is **appropriate** when the relationship between train and test mirrors the relationship between train and the deployment data. It is **biased** when it changes that relationship in one direction: too easy (test molecules sit inside training chemistry) or too hard (an artificially extreme gap). It is **nonsensical** when the axis it separates on has nothing to do with the deployment question -- splitting on Murcko scaffolds to ask whether a model transfers to a new target is of that last kind, because a scaffold says nothing about which target a molecule was made for.

Below, scenarios come first: the deployment question, then which splits answer it, which bias it, and which don't apply. Each scenario lists ✅ the right splits, ➕ useful complements, ⚠️ biased splits (with the direction of the bias), and ❌ nonsensical ones. Names are chemsplit registry ids -- see [Available splitters](#-available-splitters) for the full table of 64. A record here is a molecule, a protein target, or a molecule-target pair: mixtures, formulations, reactions and polymers are out of scope, and forcing one of those onto a molecule-level split produces a number about the wrong object.

<details>
<summary><strong>24 deployment scenarios -- click to expand</strong></summary>

### 1. Checking the pipeline works, or setting an upper bound

> "Is my featurization and training wired correctly? What is the best this model could possibly do?"

- ✅ `random`; `stratified_random` for imbalanced classes; `k_fold` or `repeated` to get a variance.
- ⚠️ Reporting this number as "generalization" is optimistic, often badly so, because near-duplicates and analogues sit on both sides.
- ⚠️ Stereoisomers, tautomers, salt forms and duplicates on both sides. Their InChIKeys all differ, so even an upper bound can score a molecule against itself -- standardise and deduplicate with `chemsplit.preprocess` before splitting.
- ❌ Any extrapolative split here. It mixes up "the pipeline is broken" with "the task is hard".

### 2. Hit identification: screening a large, diverse library

> "I will screen a vendor library or an on-demand chemical space for new actives."

Most deployment molecules are far from anything in training.

- ✅ `hi` (no test molecule above a similarity threshold to train), `similarity_threshold`, `butina`, `sphere_exclusion`, `k_means_cluster` or `opti_sim` (cluster mode) at a cutoff that matches the library's distance to your training set.
- ✅ `mood`: give it the actual library, and it picks the candidate split whose train→test distances best match train→library.
- ➕ `density_cluster` where family structure is irregular: no `k` to choose, and `noise_policy="test"` holds out the molecules that belong to no family. `projection` when an interpretable split direction matters more than cluster quality.
- ➕ `applicability_domain` (performance against distance), and `adversarial` in audit mode to confirm the split's shift is the one you intended.
- ⚠️ `murcko_scaffold`: optimistic. Molecules with different Murcko scaffolds can still be near-identical, for example after adding a ring.
- ⚠️ `spectral`, `umap_cluster`, `max_dissimilarity`: often more pessimistic than a real library. The hardest split is not the right one unless `mood` says so.
- ⚠️ `density_cluster`'s noise bucket can swallow 40% of a diverse library, and the default `noise_policy="own_groups"` scatters it, making the split quietly easier.
- ⚠️ `projection` with `method="tsne"`: t-SNE distances between clusters are an artefact of its cost function, not a chemical distance. PCA on fingerprints largely tracks molecule size.
- ❌ `max_min`/Kennard-Stone, `spxy`, `duplex`, `d_optimal`, `support_points`, `minimal_test_set_dissimilarity`, `distinct_label`, `self_organizing_map` in stratified mode. These put test molecules inside the training distribution by design, which answers the opposite question.
- ❌ `lo`: it tests ranking within known series, which is not what screening does.

### 3. Lead optimization: predicting new analogues in a known series

> "Chemists will make the next analogues of series we already have data on."

Deployment molecules are close neighbours of training molecules. What matters is ranking within a series and catching small changes with large effects.

- ✅ `temporal` within a project, using real synthesis or registration dates. This is the gold standard: it reproduces the actual design order.
- ✅ `lo`: holds out clusters of similar molecules that still span a range of activity, and scores ranking within each cluster.
- ➕ `activity_cliff`, as a diagnostic only, scored separately: are pairs with big potency jumps predicted at all?
- ➕ `simpd`, but only when real dates are missing.
- ⚠️ `random` within a project: mildly optimistic, because "future" analogues leak backwards. Acceptable as a lower bound on error.
- ❌ `hi`, `murcko_scaffold`, `butina`, `matched_molecular_series`, and every cluster split. They remove the series from training, so they ask "predict a series you've never seen", which lead optimization never does.

### 4. Benchmark design: generalisation or memorisation?

> "My model beats the baselines on this benchmark. Is that chemistry, or is it analogue bias?"

The question is not about a deployment set at all; it is about whether the benchmark's own train/test geometry hands the model the answer. A nearest-neighbour baseline that matches a trained model is the symptom.

- ✅ `ave`: minimises the asymmetric nearest-neighbour bias that lets a 1-NN lookup win a virtual screen, and reports `ave_initial` so the debiased and raw scores can be compared.
- ✅ `decoy_benchmark`: keeps each active grouped with its property-matched decoys so the pair never straddles the boundary, which is the other half of the same artefact.
- ➕ `adversarial` in audit mode, and `audit_split`'s nearest-neighbour and scaffold-overlap statistics, to quantify the bias before and after.
- ➕ `hi` or `similarity_threshold` as a contrast arm: a model whose score survives both debiasing and a similarity constraint is learning something.
- ⚠️ `ave` driven all the way to zero: over-corrects, strips genuine signal with the artefact, and is defined only for binary labels. Report the debiased and raw splits side by side.
- ⚠️ `murcko_scaffold` as a debiasing measure: it is the split most often found to be analogue-biased, not a fix for it.
- ⚠️ Stereoisomer, tautomer and salt pairs straddling the boundary. `audit_split` counts exact duplicates, but these are not exact duplicates and pass the check -- standardise first rather than relying on the audit.
- ⚠️ Property-matched decoys carry a bias of their own -- matching 2-D properties while enforcing topological dissimilarity leaves a latent signature a deep model can learn. Measure it with `ave`.
- ❌ `max_min`, `d_optimal`, `support_points`, `duplex`, `spxy`. They optimise coverage of the feature space, which neither creates nor removes analogue bias; they just change the subject.

### 5. Is method A better than method B?

> "Two models, or my model against a published number. Which one wins?"

What matters is that both methods meet the same test set, and that the gap between them is bigger than the gap between two splits of the same data.

- ✅ `predefined` to reuse a published partition exactly -- the only way to compare against a published number rather than against your own re-split.
- ✅ `repeated`, `k_fold` or `monte_carlo`, with both methods scored on *the same* splits, so the comparison is paired.
- ➕ `three_way` or `nested_cv`, so neither method is tuned on the test fold. A tuned model against an untuned baseline is not a comparison.
- ➕ `audit_split` and `ave` on an inherited partition, to see whether the number you are chasing was measured on a biased split.
- ⚠️ A conclusion from a single split. Split variance routinely exceeds the difference between two methods, so report the spread from `repeated`.
- ⚠️ `monte_carlo`: test sets overlap across repeats, so the repeats are correlated and the naive standard error is optimistic.
- ⚠️ `predefined` is keyed on row order, so filtering or re-standardising the data silently moves what the indices point to. Re-key on InChIKey.
- ❌ Re-splitting a published benchmark and comparing to its published number. Two splits of one dataset are two different experiments.
- ❌ Comparing scores from different splitters, or from different strictness. The number measures the split as much as the model.

### 6. Activity cliffs: small changes with large effects

> "Will the model notice that this one methyl costs two logs of potency?"

- ✅ `activity_cliff`, with `cliff_mask` used to score cliff and non-cliff compounds separately. The comparison is the result; the aggregate is not.
- ➕ `matched_molecular_series`: tests whether an R-group effect learned in one constant context transfers to another.
- ➕ `lo`, which scores within-cluster ranking over clusters that deliberately span an activity range.
- ⚠️ Cliff detection moves a lot with `similarity_threshold` and `fold_change_threshold`; a conclusion that survives only one setting is a fingerprint artefact, not a finding.
- ⚠️ Assay noise manufactures cliffs. A 10-fold jump between two single-shot measurements from different papers is measurement error -- aggregate replicates and prefer single-assay data first.
- ⚠️ `random`: cliff partners land on both sides, so a good aggregate score can coexist with total failure on every cliff.
- ❌ `hi`, `similarity_threshold`, `butina`, `spectral`, `max_dissimilarity`. A cliff is only a cliff if its partner is in training; remove the neighbourhood and the held-out molecule is merely out of domain, which is a different experiment.

### 7. Scaffold hopping: a new chemotype for the same pharmacophore

> "Will it rank a compound that hits the same target from a different core?"

- ✅ `scaffold_hop`: test actives whose scaffolds are absent from train, while 2-D pharmacophoric similarity to a training active is retained. This is the only splitter here that enforces both halves of the question.
- ✅ `generic_scaffold`, `scaffold_tree` at a coarse pruning level, or `ring_system` for a label-free, weaker version when actives are too few for `scaffold_hop`.
- ➕ `intersection` with a scaffold criterion as primary and a similarity grouping as secondary, so a near-identical analogue cannot cross the boundary through a scaffold technicality.
- ➕ `audit_split`'s shared-scaffold and shared-ring-system counts, which catch the technicality directly.
- ⚠️ `murcko_scaffold`: optimistic. Adding or opening a ring produces a new scaffold without producing a hop.
- ⚠️ `hi`, `max_dissimilarity`: too hard in the wrong direction. They remove the pharmacophoric relationship as well as the scaffold, which turns the question back into hit identification.
- ⚠️ 2-D pharmacophore similarity is a weak proxy for 3-D recognition; a pair that passes `min_pharm_similarity` may bind in completely different ways.
- ❌ `random`, `stratified_random`, `property`, `label_extrapolation`. None of them separates chemotypes.

### 8. A chemistry the model has never seen

> "We're moving to covalent inhibitors -- or to boron, macrocycles, PROTACs."

A new warhead changes one feature, not the whole molecule. Scaffold and fingerprint similarity to training both stay high, so neither kind of split sees the move.

- ✅ `substructure`, naming the new chemistry with `smarts`, `elements` or `functional_groups`. It is the only splitter that separates on a feature you name rather than on a distance.
- ✅ `group_k_fold` over one group label per modality, when several new chemistries are in play and each should take a turn.
- ➕ `intersection` with `substructure` as primary and a scaffold grouping as secondary, when the new modality also brings new cores.
- ⚠️ Sizes follow the matches, not your request: a rare warhead gives a tiny test set, a common element a huge one. Read `metadata["n_matched"]`.
- ⚠️ SMARTS details -- aromaticity, hydrogens, charges -- decide what matches. Inspect a few matched molecules before trusting the split.
- ⚠️ `murcko_scaffold`, `similarity_threshold` as proxies. A new warhead on a known core keeps the scaffold and most of the fingerprint, so both scatter the new chemistry across train and test.
- ⚠️ A featurizer that cannot represent the new chemistry at all -- no boron in the atom typing, a macrocycle beyond the ring perception -- fails for a reason no split can show.
- ❌ `random`, `k_fold`, `stratified_random`. They spread the new modality evenly across both sides, assuming away the question.
- ❌ `label_extrapolation`, `property`. A new modality is not a position on a label or property axis.

### 9. Prospective performance: will it still work next quarter?

> "The project's chemistry is moving. Does the model hold up on compounds that do not exist yet?"

- ✅ `temporal` on real dates, preferably `mode="rolling"` with an `embargo`, so the score is an average over several eras rather than one contiguous, chemically homogeneous block.
- ✅ `deposition_date` for structure-based models, where redundant re-depositions otherwise put near-identical complexes on both sides of the cut.
- ➕ `repeated` over rolling windows for a variance estimate, and `applicability_domain` plus `adversarial` (audit) to attribute a drop to chemical drift rather than to assay or volume changes.
- ➕ `simpd` when no usable dates exist: a GA rearranges the dataset until it reproduces the descriptor and property shifts measured in real time splits.
- ⚠️ `simpd` as a substitute for dates: it simulates the statistics of a time split, not time. Changing project goals and genuine unforeseeability are not reproduced.
- ⚠️ `temporal` confounds several shifts at once -- chemistry, protocol, target selection, data volume -- so it shows degradation, not its cause. And dates are frequently wrong: registration, first-test, publication and deposition dates differ, and datasets mix them.
- ⚠️ `butina`, `murcko_scaffold` as "proxies for time": they capture one component of drift and miss the protocol and selection components entirely.
- ❌ `random`, `k_fold`, `monte_carlo`, `stratified_distribution`, `minimal_test_set_dissimilarity`, `distinct_label`, `support_points`. Every one of them lets the future leak backwards.

### 10. Generative design and active learning

> "A generator or an acquisition function decides what we make next."

No splitter reproduces this: the deployment molecules are chosen *by* the model's own predictions and change every cycle. The splits below are approximations, and the real evaluation is the next cycle's results.

- ✅ `temporal` in `mode="rolling"` on design-cycle dates, so each window is tested on the cycle it influenced. The closest available analogue of the loop.
- ✅ `mood` re-run on each generated batch as the `deployment_set`, so the split moves as the generator does.
- ➕ `label_extrapolation` for the top of the range, where an acquisition function spends most of its time.
- ➕ `applicability_domain` to read the error at the distance the generator is proposing at, and `adversarial` (audit) per batch to see how far it has travelled.
- ⚠️ Any retrospective split reported as the loop's expected performance. The score is measured on molecules a chemist chose; the loop runs on molecules the model chose.
- ⚠️ Once predictions steer what gets made, later cycles are no longer an independent sample, and no split can undo that selection.
- ❌ `random`, `k_fold`, `monte_carlo` over pooled cycles. Later cycles exist because of earlier predictions, so pooling lets the loop leak backwards.

### 11. A different lab, vendor or assay

> "The model was trained on our data. It will be applied to someone else's."

Between-source differences are systematic offsets, not noise: the same Ki measured in two labs routinely differs by more than the model's error bar.

- ✅ `source`, at a hierarchy level that matches the deployment gap -- document-level is weaker than lab-level, assay-level weaker still.
- ✅ `external_holdout` when the other source's data is already in hand; it forces that dataset to be the entire test set.
- ➕ `leave_one_cluster_out` or `group_k_fold` over the source labels, since a single held-out source is one sample of one distribution.
- ➕ `intersection` with `source` as primary and a scaffold or similarity grouping as secondary -- grouping by provenance does **not** guarantee chemical separation, because two labs can publish the same series.
- ➕ `adversarial` in audit mode, to check the shift you obtained is the one you meant.
- ⚠️ `source` alone: a `NaN`-heavy source column degenerates silently toward a random split. Read `metadata["n_missing_source"]`.
- ⚠️ "External" describes provenance, not chemistry: a holdout drawn from the same vendor catalogue is not external in any useful sense. Check `max_similarity_to_train`.
- ⚠️ `temporal` as a proxy for provenance: eras and sources correlate, which is exactly why neither cleanly isolates the other.
- ❌ `random`, `k_fold`, `stratified_random`. They distribute every source evenly across both sides, which assumes away the entire question.
- ❌ `murcko_scaffold`, `butina` as stand-ins for provenance. Chemistry is not the axis that differs.

### 12. Cheap measurements now, expensive measurements later

> "We have thousands of single-shot or computed values and a few hundred dose-response ones. Can the first predict the second?"

- ✅ `fidelity`: train on the low-fidelity levels, test on the highest, with `structure_leakage` controlling what happens to molecules measured at both.
- ➕ `intersection` or `group_k_fold` over a scaffold or similarity grouping, so that an analogue measured at both fidelities does not reintroduce the leak that `structure_leakage` removes for exact matches.
- ➕ `source` when fidelity and assay provenance coincide, to separate the two explanations for a drop.
- ⚠️ `fidelity` confounds the label's fidelity with the chemistry measured at each level -- high-fidelity assays run on already-optimised compounds. The drop is real; its cause is not isolated.
- ⚠️ Low- and high-fidelity labels may be different quantities on different scales. The split does not harmonise them, and sizes are coarse with few levels: read `metadata["level_partition"]`.
- ❌ `random` over the pooled levels: the same molecule appears at both fidelities, so the test label is a replicate of a training label.
- ❌ `label_extrapolation`. It cuts the label *range*, not the label's provenance; the two are routinely confused.

### 13. A consortium model: does it help every contributor?

> "Several partners pool data, or train federated. Is the result useful to the small contributors or only to the largest?"

- ✅ `party` for a deliberately non-IID partition across owners, reported **per party** with `party_sizes` alongside.
- ✅ `source` whenever real owner labels exist -- real parties beat synthesised ones every time.
- ➕ `leave_one_cluster_out` or `group_k_fold` over party labels, so each partner takes a turn as the held-out one.
- ➕ A per-party `temporal` split, since each partner's chemistry drifts on its own schedule.
- ⚠️ A pooled average across parties is dominated by the largest contributor and hides a model that is useless to everyone else. This is the failure the scenario exists to detect, so never report it.
- ⚠️ `dirichlet_alpha` has no natural value and must be reported; results at `alpha=0.1` and `alpha=1.0` are not comparable. Synthesised parties model heterogeneity, not the real thing -- real partners differ in assay protocol and target selection, not just chemistry.
- ⚠️ This is a data split, not a privacy guarantee. It says nothing about what the training scheme leaks.
- ❌ `random`, `stratified_random`, `k_fold`. Pooling IID is precisely the assumption under test.
- ❌ `hi`, `murcko_scaffold` as proxies for owner boundaries. Partners are not defined by chemotype.

### 14. A multi-endpoint ADMET panel

> "One model, twenty endpoints, and most compounds measured on only a few of them."

- ✅ `balanced_multi_task`: whole clusters go to folds under a per-task constraint, so no endpoint ends up with zero test actives. Report `per_task_fold_counts`.
- ✅ `stratified_random(multitask="iterative")` or `k_fold(stratify=True, multitask="iterative")` when only the per-label balance is in question.
- ➕ Any group-forming clusterer as `balanced_multi_task`'s `clusterer`, by registry id or instance. The chemical criterion stays yours; the splitter only balances.
- ➕ `intersection` or `group_k_fold` over a scaffold grouping when the endpoints come from different assays on overlapping series.
- ⚠️ Infeasibility is normal on sparse matrices: a task with three actives in one cluster cannot be balanced. `on_infeasible="relax"` proceeds, but then the balance achieved is not the one requested.
- ⚠️ Balancing uses the labels, so the design is mildly label-aware. Disclose it.
- ⚠️ The solve is time-limited and picks its algorithm by problem size, so read `metadata["solver"]` before comparing scores across datasets.
- ⚠️ A panel average over per-task scores. It is dominated by the densest endpoints, and the sparse ones are usually why the model exists.
- ❌ `random` over a long-format `(compound, endpoint, value)` table. The same compound lands in train for one endpoint and test for another.
- ❌ `label_extrapolation`, `distinct_label` across a panel. Both act on one column at a time, so the folds differ per endpoint.

### 15. A known target, a new compound

> "Fixed panel of targets, novel chemistry against them."

- ✅ `cold_drug` **with** a `compound_grouper` (scaffold- or similarity-based), so held-out compounds are structurally novel rather than merely unseen keys.
- ➕ `balanced_multi_task` when the matrix is a multi-task problem and every task needs a workable train/test ratio.
- ➕ `intersection` to stop an analogue of a training compound slipping in under a different identifier.
- ⚠️ `cold_drug` without a grouper: optimistic. A different registry number is not different chemistry.
- ⚠️ This is the weakest of the three cold-start settings, and the most often reported as if it were the strongest: a model can score well on target-level marginals ("this kinase is promiscuous") while learning nothing about interactions.
- ⚠️ Targets can disappear from training altogether, hiding a cold-target evaluation inside a cold-drug split. Read `targets_lost_from_train`.
- ❌ `random` over interaction records: the same compound appears on both sides against different targets.
- ❌ `cold_target`, `cold_pair` as substitutes. They answer different questions and their scores are not comparable to this one.

### 16. A new target with no ligand data

> "Can the model say anything about a target we have never screened?"

- ✅ `cold_target` paired with a sequence-identity grouping on the target axis -- holding out a kinase while training on its 95%-identical paralogue is not a cold-target experiment, and the splitter warns every time the grouping is missing.
- ✅ `leave_one_cluster_out` over target groups when target counts are small, which they almost always are; holding out 20% of thirty targets gives a number with no stable error bar.
- ➕ `protein_family` to measure how far the novelty extends: a new target inside a familiar family is a much easier question than a new family.
- ➕ A different species, cell line or readout is the same question on a different axis: group by that column with `source`, or with `leave_one_cluster_out(clusterer="source")` so each takes a turn as the held-out fold.
- ⚠️ Compounds are seen, so compound-level marginals ("this compound is promiscuous") are exploitable in the same way `cold_drug` exploits target marginals.
- ⚠️ Each held-out target brings its own assay protocol, so the biological shift arrives bundled with a protocol shift.
- ❌ `random`, `cold_drug`, and any ligand-axis split -- `murcko_scaffold`, `butina`, `hi`. Separating the compound axis answers the compound question, whatever the target axis happens to do.

### 17. Both new at once

> "A new compound against a new target -- the honest worst case for an interaction model."

- ✅ `cold_pair`, with groupers on both axes (ligand similarity or scaffold on one, sequence identity on the other), so neither member of a test pair has a near-twin in training.
- ✅ `complex_joint` for structure-based models, in `mode="both_novel"`.
- ➕ `leave_one_cluster_out` on the target axis to spread the variance of a small target count.
- ⚠️ `cold_pair` discards the two off-diagonal blocks -- typically 50-90% of the data -- so the surviving test block is small and noisy, and the realised test fraction is a product of both axes that rarely matches the request.
- ⚠️ `complex_joint` routinely discards more than 80%; `mode="either_novel"` is far weaker and is frequently reported as if it were `"both_novel"`.
- ⚠️ Comparing these scores to `cold_drug` or `cold_target` numbers. Performance here is much lower by construction, and the comparison is a common, serious error.
- ❌ `random`, `cold_drug`, `cold_target` presented as this setting.

### 18. A new protein family, or a familiar sequence with an unfamiliar pocket

> "How far across target space does the model travel -- and does it recognise the pocket or the sequence?"

- ✅ `protein_family`, ideally as leave-one-family-out via `leave_one_cluster_out`, for the family question.
- ✅ `binding_site` for the pocket question, where pockets are defined: it separates on pocket residue composition rather than global sequence, so a familiar sequence with a remodelled site counts as novel.
- ➕ `sequence_identity` as a floor on both, and `complex_joint` for structure-based models that must also face novel ligands.
- ⚠️ `sequence_identity` as a proxy for pocket novelty: two proteins at 20% overall identity can share nearly identical pockets, and the split will separate them while leaking the pharmacology. Percent identity is also not one number -- local vs. global and the choice of denominator change it substantially.
- ⚠️ Family annotations are incomplete and heavily skewed (kinases dominate public data), so the requested ratio is usually unreachable and the unlabelled targets form a junk group worth checking. Family boundaries do not imply pharmacological independence either: ATP-binding proteins outside the kinase family still share ligand chemistry.
- ⚠️ `binding_site` needs a pocket definition the library cannot produce, and `residue_composition` ignores geometry -- two pockets with the same residue counts and different shapes look identical to it.
- ❌ `random`, and every ligand-axis split, for either question.

### 19. Predicting outside the measured range

> "Everything we have measured is between 10 nM and 10 µM. Will it pick the single-digit nanomolar compound?"

- ✅ `label_extrapolation`, trained on one part of the label range and tested on another, **with a ranking metric (Spearman, top-k enrichment) reported next to RMSE/R²** -- a model that predicts its training mean posts a respectable RMSE with zero rank correlation.
- ✅ `property` for a physicochemical range rather than an activity one (solubility, molecular weight, logP), noting that `direction="middle_test"` is an interpolation test living in the same family.
- ➕ `applicability_domain` to see whether the failure tracks distance or the label itself.
- ➕ `intersection` over a scaffold or similarity grouping: `property` is not a leakage control, and a test molecule can be a close analogue of a training one that happens to sit below the cut.
- ⚠️ Selecting the test set with `y` makes the experiment label-aware by construction, and the extremes are exactly where censored values, transcription errors and measurement artefacts concentrate.
- ⚠️ Metrics on a truncated label range are not comparable to the same metrics on a random split. Do not put them in one table without saying so.
- ⚠️ `property` on molecular weight: MolWt correlates with almost everything, including promiscuity, assay artefacts and the era a series was made.
- ❌ `stratified_random`, `stratified_distribution`, `distinct_label`, `minimal_test_set_dissimilarity`, `spxy`. All of them work to keep every part of the label range represented in training -- the opposite of what this scenario asks.
- ❌ `label_extrapolation` on a binary label. It degenerates into a class holdout where the model never sees a positive.

### 20. Very small datasets

> "I have eighty compounds. Which split survives that?"

Below about a hundred records every strict criterion becomes unstable: cluster splits give folds of two or three molecules, and holding out one series can remove most of the label range.

- ✅ `k_fold(n_splits="loo")`, so every record is tested and training sets stay as large as the data allows.
- ✅ `repeated` around whichever criterion you choose. At this size the spread between splits *is* the result.
- ✅ `nested_cv` if anything is tuned at all: the inner-loop optimism is larger than most effects you would be looking for.
- ➕ `group_k_fold(n_splits="auto")` for leave-one-group-out over a coarse grouping, which keeps a chemical criterion while testing every group.
- ⚠️ Leave-one-out has very high variance for classification metrics, and ROC-AUC is undefined per fold. Pool the out-of-fold predictions before scoring.
- ⚠️ `leave_one_cluster_out` with too few clusters: training sets then differ wildly between folds, so the fold scores aren't comparable.
- ⚠️ Any strict criterion here. `hi`, `butina`, `spectral` will all produce the split you asked for, with an error bar wider than the difference you are studying. Report fold sizes.
- ⚠️ A test fraction given as a percentage. Twenty per cent of eighty compounds is sixteen molecules, and no metric on sixteen molecules separates two models.
- ❌ `ave`, `simpd`. Both optimise against a statistic estimated from the data, and at this size that statistic is too noisy to optimise against.

### 21. A model built on pretrained embeddings

> "My model sits on a pretrained encoder. Is my test set held out from the encoder too?"

Two things break at once: test molecules can sit close together in the space the model actually uses even when fingerprint distances look large, and they may have been in the encoder's pretraining corpus.

- ✅ `latent_space`, splitting in the embedding the model perceives rather than a fingerprint space it never sees. Set `independence_declared=True` only when the encoder is not the model under evaluation.
- ✅ An overlap check against the pretraining set: re-key on InChIKey where the corpus is public, or build an `external_holdout` from records that postdate the checkpoint.
- ➕ `temporal` or `deposition_date` cut at the checkpoint date -- the only split that bounds pretraining contamination without the corpus in hand.
- ➕ `intersection` with `latent_space` as primary and a fingerprint grouping as secondary, so the split is strict in both spaces.
- ⚠️ Defining the split with the evaluated model's own encoder. The split is then tuned to be easy or hard for that model, and cross-model comparison stops meaning anything.
- ⚠️ Latent geometry drifts with checkpoint, tokenizer and pooling, so pin the encoder; `embedding_hash` ties a split to one artefact.
- ⚠️ Cosine distance in a latent space has no chemical units, so a cutoff carried over from Tanimoto intuition means nothing.
- ⚠️ `murcko_scaffold`, `butina`, `similarity_threshold` as evidence of novelty here. They constrain a space the model doesn't use, and say nothing about pretraining overlap.
- ❌ A `random` split presented as a holdout for a model whose encoder saw hundreds of millions of molecules.

### 22. Predicting only inside a declared applicability domain

> "The model only answers when a compound falls inside its domain."

Classical regulatory QSAR, and the one setting where putting test molecules *inside* the training distribution is correct: deployment is interpolation, because everything else is refused.

- ✅ `max_min` (MaxMin / Kennard-Stone), whose `metadata["coverage_radius"]` is the domain radius you then enforce at prediction time.
- ✅ `duplex` when both partitions must span the same space, `d_optimal` for a model linear in chosen descriptors, `support_points` when both subsets must follow the joint feature-and-label distribution.
- ✅ `minimal_test_set_dissimilarity` or `spxy` for the label-aware version: every test compound is the most typical member of its activity bin, so it always has a close training analogue.
- ✅ `stratified_distribution` when statistics must cover the full label range, and `density_cluster(noise_policy="test")` for the oddities at the domain's edge.
- ➕ `applicability_domain` to state the domain as error against distance instead of asserting it, and to choose the radius you enforce.
- ➕ `three_way`, since the domain argument applies to the validation set too.
- ⚠️ Reporting any of these as generalisation. They measure interpolation inside the domain, which is a much easier question, and the number isn't comparable to a scaffold or time split.
- ⚠️ Using them when nothing enforces the domain at prediction time. The model will then be asked about compounds no split ever evaluated it on.
- ⚠️ `spxy`, `minimal_test_set_dissimilarity` and `stratified_distribution` pick the test set using `y`, so the design is label-aware. Disclose it.
- ❌ `hi`, `max_dissimilarity`, `perimeter`, `label_extrapolation`. They measure performance outside the domain -- exactly where the model declines to predict.

### 23. My test split is strict. Is my validation split?

> "I split on scaffolds, then tuned on a random validation set. Is that a problem?"

Yes. A strict outer split with a random inner one picks hyperparameters that suit interpolation, so the test score reflects choices made on easier data than the test set.

- ✅ `three_way`, which applies the same criterion at the train/validation cut and at the validation/test cut, so the validation score previews the test score honestly.
- ✅ `nested_cv` whenever anything is tuned: each outer fold's tuning sees only that fold's training data.
- ➕ `repeated` around either, since a strict validation set is smaller and harder, which makes model selection noisier.
- ⚠️ A random validation split under a strict outer split. It is the default in most code, and it is how a strict test score gets quietly spent.
- ⚠️ `three_way` left at its default `valid_size` degenerates into a two-way split with an empty validation set. Set it, and read the realised sizes.
- ⚠️ `nested_cv` estimates the whole tuning *procedure*, not one model. The final model is refit on everything and can't inherit the nested estimate.
- ⚠️ The cost is `outer × (1 + inner)` fits. That is why it gets skipped, and skipping it is the most common silent source of optimism in published QSAR.
- ❌ Tuning on the test fold and reporting the best score seen. Nothing in the published numbers reveals it.

### 24. Does my evaluation set look like the library I will screen?

> "Before I trust any of these numbers: is my test set in the same relationship to training as my deployment library is?"

This is the applicability-domain question, and it is the one that decides which of the scenarios above you are actually in.

- ✅ `mood`, given the real deployment library: it selects, among candidate splitters, the one whose train→test distance distribution best matches train→deployment.
- ✅ `applicability_domain` for a distance-versus-performance curve rather than a single number, so the score can be read at the distance your library actually sits at.
- ➕ `external_holdout` when part of the library has been measured, and `adversarial` (audit) plus `audit_split`'s nearest-neighbour profile to describe the gap you have.
- ⚠️ `mood` needs the deployment set up front; a guessed library silently decides the answer. Its selection also uses the data, so the reported score is mildly optimistic unless the selection is disclosed. If the library overlaps training, `mood` correctly picks a random split -- which readers may mistake for a weak evaluation.
- ⚠️ `applicability_domain` bands are subsets of one test set, so each is small and noisy, and the x-axis is fingerprint- and metric-dependent. A tidy monotone curve can also be a confound: check whether molecular size increases along the bands.
- ⚠️ `max_dissimilarity`, `perimeter`, `spectral` chosen as "the conservative option". The hardest split is the right answer only if the deployment library is genuinely that far away; `perimeter` in particular holds out fragments, salts and standardisation failures, so a poor score may be a data-quality result.
- ❌ A single `random` split offered as evidence of domain coverage.

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
<summary><strong>Full list of splitters (64 classes)</strong> -- click to expand</summary>

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
| Scaffold | `SubstructureSplitter` | `substructure` | Holds out molecules containing a SMARTS pattern, element or functional group | 🚀 extrapolative | -- | -- | -- |
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
| Similarity | `MinimalTestSetDissimilaritySplitter` | `minimal_test_set_dissimilarity` | One most-typical record per activity bin goes to test (MTSD) | 🔴 optimistic | -- | labels | -- |
| Similarity | `SupportPointsSplitter` | `support_points` | Subsets nearest the energy-distance support points of the data (SPlit) | 🔴 optimistic | -- | -- | -- |
| Similarity | `DuplexSplitter` | `duplex` | Partitions take turns adding their most distant record (DUPLEX) | 🟠 moderate | -- | -- | -- |
| Similarity | `DOptimalSplitter` | `d_optimal` | Training set maximises det(XᵀX) by Fedorov exchange | 🟠 moderate | -- | -- | -- |
| Similarity | `MaxDissimilaritySplitter` | `max_dissimilarity` | Pushes train and test to opposite regions of chemical space | 🚀 extrapolative | -- | -- | -- |
| Similarity | `PerimeterSplitter` | `perimeter` | Holds out the outskirts; trains on the dense core | 🚀 extrapolative | -- | -- | -- |
| Similarity | `LeaveOneClusterOutSplitter` | `leave_one_cluster_out` | Each cluster takes a turn as the test fold | 🚀 extrapolative | :heavy_check_mark: | -- | -- |
| Similarity | `BalancedMultiTaskSplitter` | `balanced_multi_task` | Assigns whole clusters to folds, balancing every task | 🟢 strict | :heavy_check_mark: | labels | -- |
| Embedding | `UMAPClusterSplitter` | `umap_cluster` | UMAP embedding followed by clustering | 🚀 extrapolative | :heavy_check_mark: | -- | `umap` |
| Embedding | `ProjectionSplitter` | `projection` | Linear/manifold projection, then clustering, axis cut, or grid | 🟢 strict | :heavy_check_mark: | -- | -- |
| Embedding | `SelfOrganizingMapSplitter` | `self_organizing_map` | Kohonen map cells held out whole, or sampled proportionally | 🟢 strict | :heavy_check_mark: | -- | `som` |
| Embedding | `LatentSpaceSplitter` | `latent_space` | Clusters in a caller-supplied embedding | 🟢 strict | :heavy_check_mark: | -- | -- |
| Property | `PropertySplitter` | `property` | Cuts along a continuous molecular property | 🟢 strict | -- | -- | -- |
| Property | `LabelExtrapolationSplitter` | `label_extrapolation` | Trains on one part of the label range, tests on another | 🚀 extrapolative | -- | labels | -- |
| Property | `DistinctLabelSplitter` | `distinct_label` | One record per distinct label value trains; repeats are held out (SWNW) | 🔴 optimistic | -- | labels | -- |
| Property | `StratifiedDistributionSplitter` | `stratified_distribution` | Matches the full label distribution across partitions | 🔴 optimistic | -- | labels | -- |
| Property | `MOODSplitter` | `mood` | Picks the candidate split whose train→test distances best match train→deployment | 🟢 strict | -- | -- | -- |
| Property | `AdversarialSplitter` | `adversarial` | Audits or constructs a split with a train-vs-test discriminator | 🟢 strict | -- | -- | -- |
| Lineage | `TemporalSplitter` | `temporal` | Date cut: train on the past, test on the future | 🟢 strict | -- | dates | -- |
| Lineage | `SIMPDSplitter` | `simpd` | Simulated time split optimized by a genetic algorithm | 🟢 strict | -- | labels | `ga` |
| Lineage | `SourceSplitter` | `source` | Groups by provenance (document, assay, lab, vendor, ...) | 🟢 strict | :heavy_check_mark: | -- | -- |
| Lineage | `FidelitySplitter` | `fidelity` | Trains on low-fidelity levels, tests on the highest fidelity | 🚀 extrapolative | -- | fidelity levels | -- |
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

## 🔎 Finding a published method

Many published splitting methods are a splitter or an option in chemsplit:

| Method | chemsplit | Reference |
| --- | --- | --- |
| Random split | `RandomSplitter` | -- |
| Stratified random split | `StratifiedRandomSplitter` | -- |
| Multi-label iterative stratification | `StratifiedRandomSplitter(multitask="iterative")`, `"iterative_pairs"`; `KFoldSplitter(stratify=True, multitask=...)` | Sechidis et al. 2011; Szymański & Kajdanowicz 2017 |
| Kennard-Stone | `MaxMinSplitter(init="kennard_stone")` | Kennard & Stone 1969 |
| MDKS (Kennard-Stone with Mahalanobis distance) | `MaxMinSplitter(init="kennard_stone", metric="mahalanobis")` | Saptoro et al. 2012 |
| MLM (random-mutation Kennard-Stone) | `MaxMinSplitter(init="kennard_stone", swap_fraction=0.1)` | Morais et al. 2019 |
| SPXY | `SPXYSplitter` | Galvão et al. 2005 |
| M-SPXY | `SPXYSplitter(metric="mahalanobis")` | Apinantanakon et al. 2019 |
| DUPLEX | `DuplexSplitter` | Snee 1977 |
| D-optimal design | `DOptimalSplitter` | Cook & Nachtsheim 1980; de Aguiar et al. 1995 |
| SPlit (support points) | `SupportPointsSplitter` | Joseph & Vakayil 2022 |
| Minimal test set dissimilarity (MTSD) | `MinimalTestSetDissimilaritySplitter` | Martin et al. 2012 |
| SWNW | `DistinctLabelSplitter` | Li et al. 2021 |
| OptiSim | `OptiSimSplitter` | Clark 1997 |
| Sphere exclusion | `SphereExclusionSplitter` | Gobbi & Lee 2003; Golbraikh & Tropsha 2002 |
| Taylor-Butina clustering | `ButinaSplitter` | -- |
| MaxMin diversity picking | `MaxMinSplitter` | -- |
| Bemis-Murcko scaffold | `MurckoScaffoldSplitter` | -- |
| Substructure, element or functional-group holdout | `SubstructureSplitter` | -- |
| k-means clustering | `KMeansClusterSplitter` | -- |
| DBSCAN / HDBSCAN | `DensityClusterSplitter` | -- |
| Spectral clustering | `SpectralSplitter` | -- |
| Landmark spectral clustering | `SpectralSplitter(graph="landmark")` | Chen & Cai 2011 |
| Kohonen self-organizing map | `SelfOrganizingMapSplitter` | Kohonen 1982; Guha et al. 2004 |
| Target-property sort | `LabelExtrapolationSplitter` | -- |
| Molecular-weight sort | `PropertySplitter(property="MolWt")` | -- |
| Time split | `TemporalSplitter` | -- |
| Rolling / expanding time folds | `TemporalSplitter(mode="rolling")`, `"expanding"` | -- |
| Multi-fidelity split | `FidelitySplitter` | -- |

Full citations are in each class's docstring.
