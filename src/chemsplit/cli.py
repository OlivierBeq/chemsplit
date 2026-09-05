"""``python -m chemsplit`` command-line interface.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import pandas as pd

from chemsplit.exceptions import ChemSplitError, MissingDependencyError

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_VALIDATION = 2
EXIT_MISSING_DEPENDENCY = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chemsplit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split", help="Run a splitter on an input CSV and write the split as JSON.")
    p_split.add_argument("--splitter", required=True, help="Splitter id or class name (e.g. 'butina').")
    p_split.add_argument("--input", required=True, help="Path to a CSV file.")
    p_split.add_argument("--smiles-col", required=True, help="Name of the SMILES column.")
    p_split.add_argument("--label-col", default=None, help="Name of the label (y) column, if any.")
    p_split.add_argument("--train-size", type=float, default=None)
    p_split.add_argument("--valid-size", type=float, default=None)
    p_split.add_argument("--test-size", type=float, default=None)
    p_split.add_argument("--seed", type=int, default=None, dest="random_state")
    p_split.add_argument("--out", required=True, help="Path to write the split JSON to.")

    p_audit = sub.add_parser("audit", help="Audit an existing split for leakage.")
    p_audit.add_argument("--split", required=True, help="Path to a split JSON file (from `chemsplit split`).")
    p_audit.add_argument("--input", required=True, help="Path to the same CSV file the split was built from.")
    p_audit.add_argument("--smiles-col", required=True)
    p_audit.add_argument("--out", required=True, help="Path to write the audit report JSON to.")
    p_audit.add_argument("--fail-on", default=None, help="Comma-separated flag names; exit 4 if any fire.")

    p_list = sub.add_parser("list", help="List registered splitters.")
    p_list.add_argument("--family", default=None)
    p_list.add_argument("--strictness", default=None)
    p_list.add_argument("--json", action="store_true", dest="as_json")

    return parser


def _cmd_split(args: argparse.Namespace) -> int:
    from chemsplit.registry import get_splitter

    df = pd.read_csv(args.input)
    smiles = df[args.smiles_col].tolist()
    y = df[args.label_col].to_numpy() if args.label_col else None

    kwargs: dict[str, Any] = {}
    if args.train_size is not None:
        kwargs["train_size"] = args.train_size
    if args.valid_size is not None:
        kwargs["valid_size"] = args.valid_size
    if args.test_size is not None:
        kwargs["test_size"] = args.test_size
    if args.random_state is not None:
        kwargs["random_state"] = args.random_state

    splitter = get_splitter(args.splitter, **kwargs)
    results = splitter.split_result(smiles, y)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(results[0].to_json())
    print(f"wrote {args.out} ({len(results)} fold(s); first fold written)")
    return EXIT_OK


def _cmd_audit(args: argparse.Namespace) -> int:
    from chemsplit.audit import audit_split
    from chemsplit.base import SplitResult

    with open(args.split, encoding="utf-8") as fh:
        split = SplitResult.from_json(fh.read())
    df = pd.read_csv(args.input)
    smiles = df[args.smiles_col].tolist()

    report = audit_split(split, smiles)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report.to_json())
    print(f"wrote {args.out}")
    print(report.summary())

    if args.fail_on:
        wanted = {f.strip() for f in args.fail_on.split(",") if f.strip()}
        fired = set(report.flags())
        if wanted & fired:
            print(f"FAIL: flags fired: {sorted(wanted & fired)}", file=sys.stderr)
            return 4
    return EXIT_OK


def _cmd_list(args: argparse.Namespace) -> int:
    from chemsplit.registry import list_splitters

    df = list_splitters(family=args.family, strictness=args.strictness)
    if args.as_json:
        print(df.to_json(orient="records"))
    else:
        print(df.to_string(index=False))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "split":
            return _cmd_split(args)
        if args.command == "audit":
            return _cmd_audit(args)
        if args.command == "list":
            return _cmd_list(args)
        parser.error(f"unknown command {args.command!r}")
        return EXIT_UNEXPECTED  # pragma: no cover - argparse.error() exits already
    except MissingDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MISSING_DEPENDENCY
    except ChemSplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_VALIDATION
    except Exception as exc:  # noqa: BLE001 - CLI boundary, deliberately broad
        print(f"unexpected error: {exc}", file=sys.stderr)
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
