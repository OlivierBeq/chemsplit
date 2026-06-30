import numpy as np

from chemsplit._unionfind import dense_label_encode, merge_group_labels


def test_dense_label_encode_first_appearance_order():
    out = dense_label_encode(["z", "a", "z", "m", "a"])
    assert out.tolist() == [0, 1, 0, 2, 1]
    assert out.dtype == np.int64


def test_merge_group_labels_transitive_chain():
    # a: 0,1 share label X; b: 1,2 share label Y -> {0,1,2} merge into one group via record 1
    a = np.array([0, 0, 1, 2], dtype=np.int64)
    b = np.array([0, 1, 1, 2], dtype=np.int64)
    merged = merge_group_labels(a, b)
    assert merged[0] == merged[1] == merged[2]
    assert merged[3] != merged[0]


def test_merge_group_labels_no_overlap_keeps_separate():
    a = np.array([0, 1, 2, 3], dtype=np.int64)
    b = np.array([0, 1, 2, 3], dtype=np.int64)
    merged = merge_group_labels(a, b)
    assert len(set(merged.tolist())) == 4
