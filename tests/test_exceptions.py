import warnings

import pytest

from chemsplit import exceptions as ex


def test_hierarchy():
    assert issubclass(ex.ParameterError, ex.ChemSplitError)
    assert issubclass(ex.ConfigurationError, ex.ParameterError)
    assert issubclass(ex.UnknownSplitterError, ex.ParameterError)
    assert issubclass(ex.InputKindError, ex.InputError)
    assert issubclass(ex.LabelError, ex.InputError)
    assert issubclass(ex.DegenerateGroupingError, ex.InfeasibleSplitError)
    assert issubclass(ex.ConstraintUnsatisfiableError, ex.InfeasibleSplitError)
    assert issubclass(ex.EmptyPartitionError, ex.InfeasibleSplitError)
    assert issubclass(ex.ScalabilityError, ex.ChemSplitError)
    assert issubclass(ex.MissingDependencyError, ImportError)
    assert issubclass(ex.InvariantError, AssertionError)


def test_missing_dependency_message():
    err = ex.MissingDependencyError("FooSplitter", "bio")
    assert str(err) == (
        "FooSplitter requires the 'bio' extra. Install with: pip install 'chemsplit[bio]'"
    )


def test_invariant_error_message():
    err = ex.InvariantError(
        "something impossible happened", splitter_id="random", params={"x": 1}, n_records=10
    )
    text = str(err)
    assert "file an issue" in text
    assert "random" in text
    assert "n_records=10" in text


def test_warning_details_and_stacklevel():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w = ex.SizeToleranceWarning("deviation too large", details={"partition": "test"})
        ex.warn_with_details(w)
    assert len(caught) == 1
    assert caught[0].category is ex.SizeToleranceWarning
    assert caught[0].message.details == {"partition": "test"}


def test_warnings_are_chemsplit_warning():
    for cls in (
        ex.SizeToleranceWarning,
        ex.DuplicateWarning,
        ex.ParseWarning,
        ex.StandardizationWarning,
        ex.DeterminismWarning,
        ex.DegenerateClusterWarning,
        ex.SmallPartitionWarning,
        ex.CircularityWarning,
        ex.HomologyLeakWarning,
    ):
        assert issubclass(cls, ex.ChemSplitWarning)
        assert issubclass(cls, UserWarning)
