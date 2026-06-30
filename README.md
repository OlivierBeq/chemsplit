# chemsplit

A self-contained, scikit-learn-compatible Python library of dataset-splitting strategies for
cheminformatics machine learning.

`chemsplit` implements 52 splitting strategies across nine categories — baseline, scaffold,
similarity, embedding, property, lineage, task, biomolecular, and protocol splitters — behind one
coherent, deterministic API, plus a leakage-audit module and a set of reference/synthetic
datasets.

## Status

Early-stage, under active development.

## Design goals

1. **Determinism.** Same inputs + same `random_state` ⇒ byte-identical outputs, on any platform,
   any CPU count, any `n_jobs`.
2. **Explicitness.** No silent fallbacks. If a requested configuration is infeasible, raise, never
   approximate.
3. **Honesty.** Every splitter's docstring discloses its pitfalls with the same prominence as its
   advantages.
4. **Composability.** Every group-forming splitter exposes its group labels, so any grouping can
   be fed to any protocol wrapper.
5. **scikit-learn compatibility.** `split()` is drop-in usable in `cross_val_score`,
   `GridSearchCV(cv=...)`, and `cross_validate`.

## Installing

```bash
pip install chemsplit
# optional extras:
pip install "chemsplit[umap,hdbscan,ga,bio,mmpa]" # or "chemsplit[all]"
```

## License

MIT — see [LICENSE](LICENSE).
